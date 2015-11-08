package nndep;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.concurrent.MulticoreWrapper;
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toSet;

/**
 * Neural network classifier which powers a transition-based dependency
 * parser.
 *
 * This classifier is built to accept distributed-representation
 * inputs, and feeds back errors to these input layers as it learns.
 *
 * In order to train a classifier, instantiate this class using the
 * {@link #Classifier(Config, Dataset, double[][], double[][], double[], double[][], java.util.List)}
 * constructor. (The presence of a non-null dataset signals that we
 * wish to train.) After training by alternating calls to
 * {@link #computeCostFunction(int, double, double)} and,
 * {@link #takeAdaGradientStep(edu.stanford.nlp.parser.nndep.Classifier.Cost, double, double)},
 * be sure to call {@link #finalizeTraining()} in order to allow the
 * classifier to clean up resources used during training.
 *
 * @author Danqi Chen
 * @author Jon Gauthier
 */
public class StructuredPredictClassifier {
  // E: numFeatures x embeddingSize
  // W1: hiddenSize x (embeddingSize x numFeatures)
  // b1: hiddenSize
  // W2: numLabels x hiddenSize

  // Weight matrices
  private final double[][] W1, W2, E;
  private final double[] b1;

  // Global gradSaved
  private double[][] gradSaved;

  // Gradient histories
  private double[][] eg2W1, eg2W2, eg2E;
  private double[] eg2b1;

  /**
   * Pre-computed hidden layer unit activations. Each double array
   * within this data is an entire hidden layer. The sub-arrays are
   * indexed somewhat arbitrarily; in order to find hidden-layer unit
   * activations for a given feature ID, use {@link #preMap} to find
   * the proper index into this data.
   */
  private double[][] saved;

  /**
   * Describes features which should be precomputed. Each entry maps a
   * feature ID to its destined index in the saved hidden unit
   * activation data (see {@link #saved}).
   */
  private Map<Integer, Integer> preMap;

  /**
   * Initial training state is dependent on how the classifier is
   * initialized. We use this flag to determine whether calls to
   * {@link #computeCostFunction(int, double, double)}, etc. are valid.
   */
  private boolean isTraining;

  /**
   * All training examples.
   */
  private final Dataset dataset;

  /**
   * We use MulticoreWrapper to parallelize mini-batch training.
   * <p>
   * Threaded job input: partition of minibatch;
   * current weights + params
   * Threaded job output: cost value, weight gradients for partition of
   * minibatch
   */
  private final MulticoreWrapper<Pair<Collection<GlobalExample>, FeedforwardParams>, Cost> jobHandler;

  private final Config config;

  /**
   * Number of possible dependency relation labels among which this
   * classifier will choose.
   */
  private final int numLabels;
  
  private ParsingSystem system;
  
  private DependencyParser parser= null;

  /**
   * Instantiate a classifier with previously learned parameters in
   * order to perform new inference.
   *
   * @param config
   * @param E
   * @param W1
   * @param b1
   * @param W2
   * @param preComputed
   */
  public StructuredPredictClassifier(Config config, double[][] E, double[][] W1, double[] b1, double[][] W2, List<Integer> preComputed) {
    this(config, null, E, W1, b1, W2, preComputed);
  }

  /**
   * Instantiate a classifier with training data and randomly
   * initialized parameter matrices in order to begin training.
   *
   * @param config
   * @param dataset
   * @param E
   * @param W1
   * @param b1
   * @param W2
   * @param preComputed
   */
  public StructuredPredictClassifier(Config config, Dataset dataset, double[][] E, double[][] W1, double[] b1, double[][] W2,
                    List<Integer> preComputed) {
    this.config = config;
    this.dataset = dataset;

    this.E = E;
    this.W1 = W1;
    this.b1 = b1;
    this.W2 = W2;

    initGradientHistories();

    numLabels = W2.length;

    preMap = new HashMap<>();
    for (int i = 0; i < preComputed.size(); ++i)
      preMap.put(preComputed.get(i), i);

    isTraining = dataset != null;
    if (isTraining)
      jobHandler = new MulticoreWrapper<>(config.trainingThreads, new CostFunction(), false);
    else
      jobHandler = null;
  }
  
  /**
   *   set the parsing system
   */
  public void setParsingSystem(ParsingSystem system){
	  this.system = system;
  }
  
  public void setParser(DependencyParser parser){
	  this.parser = parser;
  }

  /**
   * Evaluates the training cost of a particular subset of training
   * examples given the current learned weights.
   *
   * This function will be evaluated in parallel on different data in
   * separate threads, and accesses the classifier's weights stored in
   * the outer class instance.
   *
   * Each nested class instance accumulates its own weight gradients;
   * these gradients will be merged on a main thread after all cost
   * function runs complete.
   *
   * @see #computeCostFunction(int, double, double)
   */
  private class CostFunction implements ThreadsafeProcessor<Pair<Collection<GlobalExample>, FeedforwardParams>, Cost> {

    private double[][] gradW1;
    private double[] gradb1;
    private double[][] gradW2;
    private double[][] gradE;

    @Override
    public Cost process(Pair<Collection<GlobalExample>, FeedforwardParams> input) {
      Collection<GlobalExample> examples = input.first();
      FeedforwardParams params = input.second();

      gradW1 = new double[W1.length][W1[0].length];
      gradb1 = new double[b1.length];
      gradW2 = new double[W2.length][W2[0].length];
      gradE = new double[E.length][E[0].length];
      double cost = 0.0;
      double correct = 0.0; 

      for (GlobalExample ex : examples) {
    	  
    	  CoreMap sentence = ex.sent;
    	  List<Integer>goldActs = ex.acts;
    	  
//    	  System.out.println("new examples!");
    	  
    	  /*
    	   *   Begin to decode!
    	   */
    	  int nBeam = config.nBeam;
    	  int nSentSize = sentence.get(CoreAnnotations.TokensAnnotation.class).size();
    	  int nRound = nSentSize * 2;
    	  int nActNum = system.transitions.size();
    	  boolean updateEarly = false; //whether this example execute early update
    	  ThreadLocalRandom random = ThreadLocalRandom.current();
    	 
    	  
//    	  List<DepState> beam = new ArrayList<DepState>();
    	  Configuration c = system.initialConfiguration(sentence);
    	 // if(system.canApply(c, system.transitions.get(nActNum-1))){
    	//	  system.apply(c, system.transitions.get(nActNum-1));
    	 // }
    	 // else{
    	//	  throw new RuntimeException("The first action is not SHIFT");
    	  //}
    	  
    	  // only store the best beam candidates in decoding!
    	  DepState initialState = new DepState(c, -1, 0.0);
    	  Beam beam = new Beam(nBeam);
    	  beam.insert(initialState);

    	  // the lattice to store states to be sorted
    	  ArrayList<ArrayList<DepState>> beamLattice = new ArrayList<ArrayList<DepState>>();
        
    	  // begin to do nRound-th action
    	  for(int i = 0; i < nRound; i++){
    		  
//    		  System.err.println("round###################################");
    		  beamLattice.clear();
    		  int goldAct = goldActs.get(i);
    		  
    		  //begin to expand
    		  boolean beamLatticeGold = false;
    		  for(DepState beam_j : beam){
    			  int[] dropOutArray = IntStream.range(0, config.hiddenSize)
                          .filter(n -> random.nextDouble() > params.getDropOutProb())
                          .toArray();
    			  int[] featureArray = parser.getFeatureArray( beam_j.c );
    			  double[] scores = computeScoresInTraining(featureArray, preMap, dropOutArray);
    			  
    			  // do softmax
    			  //but in sentence level log-likelihood, we do not do softmax in every step!
    			  //so the softmax is only return the label array!
    			  List<Integer> predictLabel = parser.softmax(scores, beam_j.c);
    			  
    			  beam_j.setLabel(predictLabel);
    			  beam_j.setDropOutArray(dropOutArray);
    			  beam_j.setFeatureArray(featureArray);
    			  
    			  // add all expanded candidates to lattice
//    			  System.err.println(j+" lattice###################################");
    			  ArrayList<DepState> beamItem = new ArrayList<DepState>();  // states in beamItem are
    			  															 // expanded from the same state 
    			  for(int k = 0; k<nActNum; k++){
    				  if( predictLabel.get(k) != -1 ){
    					  DepState expandState = new DepState(beam_j.c, k , beam_j.score + scores[k], beam_j, 
    							  beam_j.bGold && k==goldAct );
    					  beamItem.add(expandState);
    					  beamLatticeGold = beamLatticeGold || expandState.bGold;
//    					  System.err.println(k+"# "+lattice.get(lattice.size()-1));
    				  }
    			  }
    			  
    			  if(beamItem.size() != 0)
    				  beamLattice.add(beamItem);
    		  }
    		  
//    		  if(!beamLatticeGold)
//    			  System.out.println("");
    		  
    		  //add from lattice to beam
    		  beam = new Beam(nBeam);
    		  for(ArrayList<DepState> beamLatticeItem : beamLattice)
    			  for(DepState state : beamLatticeItem)
    				  beam.insert(state);
    		  
    		  // apply these states, lazy expand
    		  for(DepState state : beam){
    			  state.StateApply(system);
    			  //print for debug
//    			  System.err.println(nRound+"\t"+beam_index++ +"\t"+state.toString());
//    			  System.err.println(state.actionSequence());
    		  }
    		  
    		  //early update
    		  if(!beam.beamGold()){
    				  updateEarly = true;
    				  break;
    		   }
    		  
    	  } //end nRound
    	  
    	  DepState predictState = beam.maxScoreState;
    	  List<Integer> predictActs = predictState.actionSequence();
    	  
    	  //get the first disagreement  of two action sequences
//    	  int firstDisAgreePos = -1;
//    	  for(int i = 0; i < predictActs.size(); i++){
//    		  if(goldActs.get(i) != predictActs.get(i) ){
//    			  firstDisAgreePos = i;
//    			  break;
//    		  }
//    	  }
    	  
    	  correct += ((double)predictActs.size()/nSentSize/params.batchSize)/2;
    	  
    	  /*
    	   *   if two actions sequence is the same, do not update!
    	   *   #TODO but we could choose max-margin loss, and always update!
    	   */
//    	  if(firstDisAgreePos == -1)
//    		  continue;
    	  
    	  /*
    	   *   print action sequence
    	   */
//    	  System.err.println("gold action sequence: "+goldActs);
//    	  System.err.println("predict action sequence: "+predictActs);

    	  /*
    	   *   Begin to train!
    	   */
    	  
    	  //softmax the whole beam candidates!
    	  
    	  
    	  // the parameter for training
		  if(!config.bAggressiveUpdate && !updateEarly){
			  if(beam.maxScoreState.bGold)
				  continue;  //skip update if predict right!
		  }
	
		  double[][] gradients = new double[beam.size()][nActNum];
		  double[] totalGradients = new double[beam.size()];
		  
		  double sum =0;
		  double maxVal = beam.maxScoreState.score;  //get the max score
		  for(int i = 0; i < beamLattice.size(); i++){
			  ArrayList<DepState> beamLatticeItem = beamLattice.get(i);
			  
			  for(int j = 0; j < beamLatticeItem.size(); j++){
				  int act = beamLatticeItem.get(j).act;
				  gradients[i][act] =  Math.exp(beamLatticeItem.get(j).score - maxVal);
				  sum += gradients[i][act];
			  }
		  }
		  
		  /*
		   * copy a un-normalized gradients arrays
		   */
		  double[][] gradientsUnNormal = new double[beam.size()][nActNum];
		  for(int gi = 0; gi < gradients.length; gi++)
			  gradientsUnNormal[gi] = Arrays.copyOf(gradients[gi], gradients[gi].length);
		  
		  for(int i = 0; i < beamLattice.size(); i++){
			  ArrayList<DepState> beamLatticeItem = beamLattice.get(i);
			  for(int j = 0; j < beamLatticeItem.size(); j++){
				  int act = beamLatticeItem.get(j).act;
				  
				  int t = beamLatticeItem.get(j).bGold ? 1 : 0;
//				  if(beamLatticeItem.get(j).bGold){
//					  System.out.println("i: "+i+" j: "+j);
//					  System.out.println("act: "+act);
//				  }
				 
				  gradients[i][act] =  gradients[i][act] / sum;
				  if(gradients[i][act] <= 0.5)
					  gradients[i][act]  = gradients[i][act]  - t;
				  else
					  gradients[i][act] = (1 - t) - (sum - gradientsUnNormal[i][act])/sum;
				  
				  totalGradients[i] += gradients[i][act];
			  }
		  }
		  
		  	  // update parameters
    		  /*
    		   *   training k-best candidates in the beam
    		   */
    		  for(int k = 0; k<beamLattice.size(); k++){
    			  DepState beamState = beamLattice.get(k).get(0);
    			  
    			  for(int i = 0; i<predictActs.size(); i++){
    			  
    				  //get right predict label
    				  if(beamState.act == -1)
    					  throw new RuntimeException("The action of current state is -1, the initial state!");
    				  
    				  List<Integer> label = beamState.lastState.labels;
    				  label.set(beamState.act, 1);
    				  //update predict
    				  
    				  if(i == 0)
    					  trainFeatures(params, beamState.lastState.featureArray, label, 
    						  false, 0, beamState.lastState.dropOutArray, gradients[k]);
    				  else
    					  trainFeatures(params, beamState.lastState.featureArray, label, 
        						  false, totalGradients[k], beamState.lastState.dropOutArray, null);
    				  
    				  //set the label back for next use!
    				  label.set(beamState.act, 0);
    				  beamState = beamState.lastState;
    		  }
    	  }
        
      }	//end foreach examples

      return new Cost(cost, correct, gradW1, gradb1, gradW2, gradE);
    }
    
    /**
     *   Train a feature with feedforward and back propogation
     * @param params
     * @param feature
     * @param bGold
     */
    public void trainFeatures( FeedforwardParams params, int[] feature, List<Integer> label, 
    		boolean bGold, double expDecay, int[] dropOutArray, double[] gradients){
    	
    	double[] scores = new double[numLabels];
        double[] hidden = new double[config.hiddenSize];
        double[] hidden3 = new double[config.hiddenSize];
        
        // We can't fix the seed used with ThreadLocalRandom
        // TODO: Is this a serious problem?

        // Run dropout: randomly drop some hidden-layer units. `ls`
        // contains the indices of those units which are still active
        int[] ls = dropOutArray;

        int offset = 0;
        for (int j = 0; j < config.numTokens; ++j) {
          int tok = feature[j];
          int index = tok * config.numTokens + j;

          if (preMap.containsKey(index)) {
            // Unit activations for this input feature value have been
            // precomputed
            int id = preMap.get(index);

            // Only extract activations for those nodes which are still
            // activated (`ls`)
            for (int nodeIndex : ls)
              hidden[nodeIndex] += saved[id][nodeIndex];
          } else {
            for (int nodeIndex : ls) {
              for (int k = 0; k < config.embeddingSize; ++k)
                hidden[nodeIndex] += W1[nodeIndex][offset + k] * E[tok][k];
            }
          }
          offset += config.embeddingSize;
        }

        // Add bias term and apply activation function
        for (int nodeIndex : ls) {
          hidden[nodeIndex] += b1[nodeIndex];
          hidden3[nodeIndex] = Math.pow(hidden[nodeIndex], 3);
        }

        /* 
         * in training, we only need to get the final score, we only need to 
         * get the hidden and compute gradients
         * 
         */
//        // Feed forward to softmax layer (no activation yet)
//        for (int i = 0; i < numLabels; ++i) {
//          if (label.get(i) >= 0) {
//            for (int nodeIndex : ls)
//              scores[i] += W2[i][nodeIndex] * hidden3[nodeIndex];
//          }
//        }
        
        /*
         *   get the error array!
         */
        double[] gradHidden3 = new double[config.hiddenSize];
        for (int i = 0; i < numLabels; ++i)
          if ( (gradients != null && label.get(i) != -1) || label.get(i) == 1) {
        	  
        	  double delta =  (gradients != null ? gradients[i] : expDecay) / params.getBatchSize(); 
        	  //cross entropy loss
            for (int nodeIndex : ls) {
              gradW2[i][nodeIndex] += delta * hidden3[nodeIndex];
              gradHidden3[nodeIndex] += delta * W2[i][nodeIndex];
            }
          }
          

        double[] gradHidden = new double[config.hiddenSize];
        for (int nodeIndex : ls) {
          gradHidden[nodeIndex] = gradHidden3[nodeIndex] * 3 * hidden[nodeIndex] * hidden[nodeIndex];
          gradb1[nodeIndex] += gradHidden3[nodeIndex];
        }

        offset = 0;
        for (int j = 0; j < config.numTokens; ++j) {
          int tok = feature[j];
          int index = tok * config.numTokens + j;
          if (preMap.containsKey(index)) {
            int id = preMap.get(index);
            for (int nodeIndex : ls){
              gradSaved[id][nodeIndex] += gradHidden[nodeIndex];
            }
          } else {
            for (int nodeIndex : ls) {
              for (int k = 0; k < config.embeddingSize; ++k) {
            	  gradW1[nodeIndex][offset + k] += gradHidden[nodeIndex] * E[tok][k];
                  gradE[tok][k] += gradHidden[nodeIndex] * W1[nodeIndex][offset + k];
              }
            }
          }
          offset += config.embeddingSize;
        }
    }
    
    /**
     * Return a new threadsafe instance.
     */
    @Override
    public ThreadsafeProcessor<Pair<Collection<GlobalExample>, FeedforwardParams>, Cost> newInstance() {
      return new CostFunction();
    }
  }

  /**
   * Describes the parameters for a particular invocation of a cost
   * function.
   */
  private static class FeedforwardParams {

    /**
     * Size of the entire mini-batch (not just the chunk that might be
     * fed-forward at this moment).
     */
    private final int batchSize;

    private final double dropOutProb;

    private FeedforwardParams(int batchSize, double dropOutProb) {
      this.batchSize = batchSize;
      this.dropOutProb = dropOutProb;
    }

    public int getBatchSize() {
      return batchSize;
    }

    public double getDropOutProb() {
      return dropOutProb;
    }

  }

  /**
   * Describes the result of feedforward + backpropagation through
   * the neural network for the batch provided to a `CostFunction.`
   * <p>
   * The members of this class represent weight deltas computed by
   * backpropagation.
   *
   * @see StructuredPredictClassifier.CostFunction
   */
  public class Cost {

    private double cost;

    // Percent of training examples predicted correctly
    private double percentCorrect;

    // Weight deltas
    private double[][] gradW1;
    private double[] gradb1;
    private double[][] gradW2;
    private double[][] gradE;

    private Cost(double cost, double percentCorrect, double[][] gradW1, double[] gradb1, double[][] gradW2,
                 double[][] gradE) {
      this.cost = cost;
      this.percentCorrect = percentCorrect;

      this.gradW1 = gradW1;
      this.gradb1 = gradb1;
      this.gradW2 = gradW2;
      this.gradE = gradE;
    }
    
    public void initGradients(){
    	this.gradW1 = new double[W1.length][W1[0].length];
	    gradb1 = new double[b1.length];
	    gradW2 = new double[W2.length][W2[0].length];
	    gradE = new double[E.length][E[0].length];
    }

    /**
     * Merge the given {@code Cost} data with the data in this
     * instance.
     *
     * @param otherCost
     */
    public void merge(Cost otherCost) {
      this.cost += otherCost.getCost();
      this.percentCorrect += otherCost.getPercentCorrect();

      addInPlace(gradW1, otherCost.getGradW1());
      addInPlace(gradb1, otherCost.getGradb1());
      addInPlace(gradW2, otherCost.getGradW2());
      addInPlace(gradE, otherCost.getGradE());
    }

    /**
     * Backpropagate gradient values from gradSaved into the gradients
     * for the E vectors that generated them.
     *
     * @param featuresSeen Feature IDs observed during training for
     *                     which gradSaved values need to be backprop'd
     *                     into gradE
     */
    private void backpropSaved(Set<Integer> featuresSeen) {
      for (int x : featuresSeen) {
        int mapX = preMap.get(x);
        int tok = x / config.numTokens;
        int offset = (x % config.numTokens) * config.embeddingSize;
        for (int j = 0; j < config.hiddenSize; ++j) {
          double delta = gradSaved[mapX][j];
          for (int k = 0; k < config.embeddingSize; ++k) {
            gradW1[j][offset + k] += delta * E[tok][k];
            
            if(Double.isInfinite(gradW1[j][offset + k] ) || Double.isNaN(gradW1[j][offset + k] ) ){
            	System.err.println(gradW1[j][offset + k]);
            	throw new RuntimeException("Infinite or Nan!");
            }
            gradE[tok][k] += delta * W1[j][offset + k];
          }
        }
      }
    }

    /**
     * Add L2 regularization cost to the gradients associated with this
     * instance.
     */
    private void addL2Regularization(double regularizationWeight) {
      for (int i = 0; i < W1.length; ++i) {
        for (int j = 0; j < W1[i].length; ++j) {
          cost += regularizationWeight * W1[i][j] * W1[i][j] / 2.0;
          gradW1[i][j] += regularizationWeight * W1[i][j];
        }
      }

      for (int i = 0; i < b1.length; ++i) {
        cost += regularizationWeight * b1[i] * b1[i] / 2.0;
        gradb1[i] += regularizationWeight * b1[i];
      }

      for (int i = 0; i < W2.length; ++i) {
        for (int j = 0; j < W2[i].length; ++j) {
          cost += regularizationWeight * W2[i][j] * W2[i][j] / 2.0;
          gradW2[i][j] += regularizationWeight * W2[i][j];
        }
      }

      for (int i = 0; i < E.length; ++i) {
        for (int j = 0; j < E[i].length; ++j) {
          cost += regularizationWeight * E[i][j] * E[i][j] / 2.0;
          gradE[i][j] += regularizationWeight * E[i][j];
        }
      }
    }

    public double getCost() {
      return cost;
    }

    public double getPercentCorrect() {
      return percentCorrect;
    }

    public double[][] getGradW1() {
      return gradW1;
    }

    public double[] getGradb1() {
      return gradb1;
    }

    public double[][] getGradW2() {
      return gradW2;
    }

    public double[][] getGradE() {
      return gradE;
    }

  }

  /**
   * Determine the feature IDs which need to be pre-computed for
   * training with these examples.
   */
  private Set<Integer> getToPreCompute(List<Example> examples) {
    Set<Integer> featureIDs = new HashSet<>();
    for (Example ex : examples) {
      List<Integer> feature = ex.getFeature();

      for (int j = 0; j < config.numTokens; j++) {
        int tok = feature.get(j);
        int index = tok * config.numTokens + j;
        if (preMap.containsKey(index))
          featureIDs.add(index);
      }
    }

    double percentagePreComputed = featureIDs.size() / (float) config.numPreComputed;
    System.err.printf("Percent actually necessary to pre-compute: %f%%%n", percentagePreComputed * 100);

    return featureIDs;
  }

  /**
   * Determine the total cost on the dataset associated with this
   * classifier using the current learned parameters. This cost is
   * evaluated using mini-batch adaptive gradient descent.
   *
   * This method launches multiple threads, each of which evaluates
   * training cost on a partition of the mini-batch.
   *
   * @param batchSize
   * @param regParameter Regularization parameter (lambda)
   * @param dropOutProb Drop-out probability. Hidden-layer units in the
   *                    neural network will be randomly turned off
   *                    while training a particular example with this
   *                    probability.
   * @return A {@link edu.stanford.nlp.parser.nndep.Classifier.Cost}
   *         object which describes the total cost of the given
   *         weights, and includes gradients to be used for further
   *         training
   */
  public Cost computeCostFunction(int batchSize, double regParameter, double dropOutProb) {
    validateTraining();

//    List<Example> examples = Util.getRandomSubList(dataset.examples, batchSize);
//
//    // Redo precomputations for only those features which are triggered
//    // by examples in this mini-batch.
//    Set<Integer> toPreCompute = getToPreCompute(examples);
//    preCompute(toPreCompute);
//
//    // Set up parameters for feedforward
//    FeedforwardParams params = new FeedforwardParams(batchSize, dropOutProb);
//
//    // Zero out saved-embedding gradients
//    gradSaved = new double[preMap.size()][config.hiddenSize];
//
//    int numChunks = config.trainingThreads;
//    List<Collection<Example>> chunks = CollectionUtils.partitionIntoFolds(examples, numChunks);
//
//    // Submit chunks for processing on separate threads
//    for (Collection<Example> chunk : chunks)
//      jobHandler.put(new Pair<>(chunk, params));
//    jobHandler.join(false);
//
//    // Join costs from each chunk
//    Cost cost = null;
//    while (jobHandler.peek()) {
//      Cost otherCost = jobHandler.poll();
//
//      if (cost == null)
//        cost = otherCost;
//      else
//        cost.merge(otherCost);
//    }
//
//    if (cost == null)
//      return null;
//
//    // Backpropagate gradients on saved pre-computed values to actual
//    // embeddings
//    cost.backpropSaved(toPreCompute);
//
//    cost.addL2Regularization(regParameter);
//
//    return cost;
    return null;  //for quick program, I anotate all these codes in this function
  }
  
  /**
   *   feedforward and update the parameter of other batch directly like in 
   *   {@link @link edu.stanford.nlp.parser.nndep.Classifier.Cost.computeCostFunction()}
   *   
   *   it is something like a SGD learning but with a small batch! After update batch gradients,
   *   We just make the update immediately after the decoding.
   *   
   *   NOTE: with SGD, the precomputed must be false when training!
   */
  public void computeCostFunctionAndUpdate(int batchSize, double regParameter, double dropOutProb, 
		  double adaAlpha, double adaEps){
	  
	  List<Example> examples = dataset.examples;
	  //List<Example> examples = Util.getRandomSubList(dataset.examples, batchSize);

	    // Set up parameters for feedforward
	    FeedforwardParams params = new FeedforwardParams(batchSize, dropOutProb);

	    int numChunks = config.trainingThreads;
	    List<Collection<Example>> chunks = CollectionUtils.partitionIntoFolds(examples, numChunks);

	    /*
	     *   Construct a Cost object for object reuse in our function.
	     *   Because I need a Cost object in every time calling FeedForwardAndBP
	     * 
	     */
	    double[][]  gradW1 = new double[W1.length][W1[0].length];
	    double[]   gradb1 = new double[b1.length];
	    double[][]   gradW2 = new double[W2.length][W2[0].length];
	    double[][]   gradE = new double[E.length][E[0].length];
	    Cost cost = new Cost(0, 0, gradW1, gradb1, gradW2, gradE);
	    
	    // Submit chunks for processing on separate threads
	    for (Collection<Example> chunk : chunks)
	      FeedForwardAndBP(cost, chunk, batchSize, dropOutProb, regParameter, adaAlpha, adaEps);

  }
  
/**
 *   Feedforward weights and update them immediately
 * 
 * @param batchSize
 * @param dropOutProb
 * @param regParameter
 * @param adaAlpha
 * @param adaEps
 */
  private void FeedForwardAndBP(Cost retval,Collection<Example> examples, int batchSize, double dropOutProb,
		double regParameter, double adaAlpha, double adaEps) {

	  /*
	   *   Feedforward weights
	   */

      // We can't fix the seed used with ThreadLocalRandom
      // TODO: Is this a serious problem?
      ThreadLocalRandom random = ThreadLocalRandom.current();
      
      retval.initGradients();

      double[][] gradW1 = retval.gradW1;
      double[] gradb1 = retval.gradb1;
      double[][] gradW2 = retval.gradW2;
      double[][] gradE = retval.gradE;

      double cost = 0.0;
      double correct = 0.0;

      int insnum = 0;
      System.err.println("example num:" + examples.size()); 
      for (Example ex : examples) {
        List<Integer> feature = ex.getFeature();
        List<Integer> label = ex.getLabel();
        
        if( (insnum++)%1000== 0){
        	System.err.println("example:" + insnum);        	
        }

        double[] scores = new double[numLabels];
        double[] hidden = new double[config.hiddenSize];
        double[] hidden3 = new double[config.hiddenSize];

        // Run dropout: randomly drop some hidden-layer units. `ls`
        // contains the indices of those units which are still active
        int[] ls = IntStream.range(0, config.hiddenSize)
                            .filter(n -> random.nextDouble() > dropOutProb)
                            .toArray();

        int offset = 0;
        for (int j = 0; j < config.numTokens; ++j) {
          int tok = feature.get(j);

           for (int nodeIndex : ls) {
              for (int k = 0; k < config.embeddingSize; ++k)
                hidden[nodeIndex] += W1[nodeIndex][offset + k] * E[tok][k];
            }
          
          offset += config.embeddingSize;
        }

        // Add bias term and apply activation function
        for (int nodeIndex : ls) {
          hidden[nodeIndex] += b1[nodeIndex];
          hidden3[nodeIndex] = Math.pow(hidden[nodeIndex], 3);
        }

        // Feed forward to softmax layer (no activation yet)
        int optLabel = -1;
        for (int i = 0; i < numLabels; ++i) {
          if (label.get(i) >= 0) {
            for (int nodeIndex : ls)
              scores[i] += W2[i][nodeIndex] * hidden3[nodeIndex];

            if (optLabel < 0 || scores[i] > scores[optLabel])
              optLabel = i;
          }
        }

        /*
         *   Get the best label!
         */
        double sum1 = 0.0;	//sum of scores of action label == 1, in our task, only one action score will be 1, Maybe #TODO 
        double sum2 = 0.0;	//sum of scores of all actions after softmax  
        double maxScore = scores[optLabel];
        for (int i = 0; i < numLabels; ++i) {
          if (label.get(i) >= 0) {
        	  
        	  /*
        	   *   #MODIFY 
        	   *   I remove the softmax of output layer
        	   * 
        	   */
           // scores[i] = Math.exp(scores[i] - maxScore);
            if (label.get(i) == 1) sum1 += scores[i];
            sum2 += scores[i];
          }
        }

        cost += (Math.log(sum2) - Math.log(sum1)) / batchSize;
        if (label.get(optLabel) == 1){
        	correct += +1.0 / batchSize;
        	
        	/*
        	 *   #MODIFY
        	 *   I make the code directly return if the prediction is right!
        	 * 
        	 */
        	continue;    //if right? Directly return!
        	//return new Cost(cost, correct, gradW1, gradb1, gradW2, gradE);        	
        }

        /*
         *   Begin to back propagation errors!
         *   In SGD, we do the regularization in the bp errors process!
         */
        /*
         *   #MODIFY
         *   I change the code with margin loss and only update the weight 
         *   of output neurons which is golden or mis-predicted label, other
         *   than all neurons in output layer. 
         * 
         */
        double[] gradHidden3 = new double[config.hiddenSize];
        for (int i = 0; i < numLabels; ++i)
          if (label.get(i) >= 0) {
        	  double delta;
        	 if(i == optLabel) delta= 1.0;
        	 else if( label.get(i) == 1) delta=-1.0;
        	 else continue;
            //double delta = -(label.get(i) - scores[i] / sum2) / batchSize;
            for (int nodeIndex : ls) {
              gradW2[i][nodeIndex] += delta * hidden3[nodeIndex] + regParameter * W2[i][nodeIndex];
              gradHidden3[nodeIndex] += delta * W2[i][nodeIndex];
            }
          }

        double[] gradHidden = new double[config.hiddenSize];
        for (int nodeIndex : ls) {
          gradHidden[nodeIndex] = gradHidden3[nodeIndex] * 3 * hidden[nodeIndex] * hidden[nodeIndex];
          gradb1[nodeIndex] += gradHidden3[nodeIndex] + regParameter * b1[nodeIndex];
        }

        offset = 0;
        for (int j = 0; j < config.numTokens; ++j) {
          int tok = feature.get(j);
  
            for (int nodeIndex : ls) {
              for (int k = 0; k < config.embeddingSize; ++k) {
                gradW1[nodeIndex][offset + k] += gradHidden[nodeIndex] * E[tok][k];
                gradE[tok][k] += gradHidden[nodeIndex] * W1[nodeIndex][offset + k] + regParameter * E[tok][k];
              }
            }
          
          offset += config.embeddingSize;
        }
        
        /*
         *   Begin to update error!
         */
        takeAdaGradientStep(retval, adaAlpha, adaEps);
      }
      
      System.err.println("correct: "+correct/examples.size());
}

/**
   * Update classifier weights using the given training cost
   * information.
   *
   * @param cost Cost information as returned by
   *             {@link #computeCostFunction(int, double, double)}.
   * @param adaAlpha Global AdaGrad learning rate
   * @param adaEps Epsilon value for numerical stability in AdaGrad's
   *               division
   */
  public void takeAdaGradientStep(Cost cost, double adaAlpha, double adaEps) {
    validateTraining();

    double[][] gradW1 = cost.getGradW1(), gradW2 = cost.getGradW2(),
        gradE = cost.getGradE();
    double[] gradb1 = cost.getGradb1();
    
    

    for (int i = 0; i < W1.length; ++i) {
      for (int j = 0; j < W1[i].length; ++j) {
        eg2W1[i][j] += gradW1[i][j] * gradW1[i][j];
        if(Double.isInfinite(gradW1[i][j]) || Double.isNaN(gradW1[i][j])){
        	throw new RuntimeException("Infinite or Nan gradW1!");
        	//System.err.println(gradW1[i][j]) );
        }
        W1[i][j] -= adaAlpha * gradW1[i][j] / Math.sqrt(eg2W1[i][j] + adaEps);
      }
    }

    for (int i = 0; i < b1.length; ++i) {
      eg2b1[i] += gradb1[i] * gradb1[i];
      b1[i] -= adaAlpha * gradb1[i] / Math.sqrt(eg2b1[i] + adaEps);
    }

    for (int i = 0; i < W2.length; ++i) {
      for (int j = 0; j < W2[i].length; ++j) {
        eg2W2[i][j] += gradW2[i][j] * gradW2[i][j];
        W2[i][j] -= adaAlpha * gradW2[i][j] / Math.sqrt(eg2W2[i][j] + adaEps);
      }
    }

    for (int i = 0; i < E.length; ++i) {
      for (int j = 0; j < E[i].length; ++j) {
        eg2E[i][j] += gradE[i][j] * gradE[i][j];
        E[i][j] -= adaAlpha * gradE[i][j] / Math.sqrt(eg2E[i][j] + adaEps);
      }
    }
  }

  private void initGradientHistories() {
    eg2E = new double[E.length][E[0].length];
    eg2W1 = new double[W1.length][W1[0].length];
    eg2b1 = new double[b1.length];
    eg2W2 = new double[W2.length][W2[0].length];
  }

  /**
   * Clear all gradient histories used for AdaGrad training.
   *
   * @throws java.lang.IllegalStateException If not training
   */
  public void clearGradientHistories() {
    validateTraining();
    initGradientHistories();
  }

  private void validateTraining() {
    if (!isTraining)
      throw new IllegalStateException("Not training, or training was already finalized");
  }

  /**
   * Finish training this classifier; prepare for a shutdown.
   */
  public void finalizeTraining() {
    validateTraining();

    // Destroy threadpool
    jobHandler.join(true);

    isTraining = false;
  }

  /**
   * @see #preCompute(java.util.Set)
   */
  public void preCompute() {
    // If no features are specified, pre-compute all of them (which fit
    // into a `saved` array of size `config.numPreComputed`)
    Set<Integer> keys = preMap.entrySet().stream()
                              .filter(e -> e.getValue() < config.numPreComputed)
                              .map(Map.Entry::getKey)
                              .collect(toSet());
    preCompute(keys);
  }

  /**
   * Pre-compute hidden layer activations for some set of possible
   * feature inputs.
   *
   * @param toPreCompute Set of feature IDs for which hidden layer
   *                     activations should be precomputed
   */
  public void preCompute(Set<Integer> toPreCompute) {
    long startTime = System.currentTimeMillis();

    // NB: It'd make sense to just make the first dimension of this
    // array the same size as `toPreCompute`, then recalculate all
    // `preMap` indices to map into this denser array. But this
    // actually hurt training performance! (See experiments with
    // "smallMap.")
    saved = new double[preMap.size()][config.hiddenSize];

    for (int x : toPreCompute) {
      int mapX = preMap.get(x);
      int tok = x / config.numTokens;
      int pos = x % config.numTokens;
      for (int j = 0; j < config.hiddenSize; ++j)
        for (int k = 0; k < config.embeddingSize; ++k)
          saved[mapX][j] += W1[j][pos * config.embeddingSize + k] * E[tok][k];
    }
    System.err.println("PreComputed " + toPreCompute.size() + ", Elapsed Time: " + (System
        .currentTimeMillis() - startTime) / 1000.0 + " (s)");
  }

  double[] computeScores(int[] feature) {
    return computeScores(feature, preMap);
  }
  
  /**
   *   ComputerScoresInTraining
   *   
   *   In training, we need to decode with dropout, so the
   *   drop out result is computed before computer scores in this 
   *   function! 
   *   The drop out array will be used in update parameters!
   *   
   */
  private double[] computeScoresInTraining(int[] feature, Map<Integer, Integer> preMap, 
		  int[] dropOutArray){
	  
	  int[] ls = dropOutArray;
	  double[] hidden = new double[config.hiddenSize];
	  int offset = 0;
      for (int j = 0; j < config.numTokens; ++j) {
        int tok = feature[j];
        int index = tok * config.numTokens + j;

        if (preMap.containsKey(index)) {
          int id = preMap.get(index);

          // Only extract activations for those nodes which are still
          // activated (`ls`)
          for (int nodeIndex : ls)
            hidden[nodeIndex] += saved[id][nodeIndex];
        } else {
          for (int nodeIndex : ls) {
            for (int k = 0; k < config.embeddingSize; ++k)
              hidden[nodeIndex] += W1[nodeIndex][offset + k] * E[tok][k];
          }
        }
        offset += config.embeddingSize;
      }

      // Add bias term and apply activation function
      for (int nodeIndex : ls) {
        hidden[nodeIndex] += b1[nodeIndex];
        hidden[nodeIndex] = hidden[nodeIndex] * hidden[nodeIndex] * hidden[nodeIndex];  // cube nonlinearity
      }

      // Feed forward to softmax layer (no activation yet)
      double[] scores = new double[numLabels];
      for (int i = 0; i < numLabels; ++i) {
          for (int nodeIndex : ls)
            scores[i] += W2[i][nodeIndex] * hidden[nodeIndex];
      }
      
      return scores;
  }

  /**
   * Feed a feature vector forward through the network. Returns the
   * values of the output layer.
   */
  private double[] computeScores(int[] feature, Map<Integer, Integer> preMap) {
    double[] hidden = new double[config.hiddenSize];
    int offset = 0;
    for (int j = 0; j < feature.length; ++j) {
      int tok = feature[j];
      int index = tok * config.numTokens + j;

      if (preMap.containsKey(index)) {
        int id = preMap.get(index);
        for (int i = 0; i < config.hiddenSize; ++i)
          hidden[i] += saved[id][i];
      } else {
        for (int i = 0; i < config.hiddenSize; ++i)
          for (int k = 0; k < config.embeddingSize; ++k)
            hidden[i] += W1[i][offset + k] * E[tok][k];
      }
      offset += config.embeddingSize;
    }

    for (int i = 0; i < config.hiddenSize; ++i) {
      hidden[i] += b1[i];
      hidden[i] = hidden[i] * hidden[i] * hidden[i];  // cube nonlinearity
    }

    double[] scores = new double[numLabels];
    for (int i = 0; i < numLabels; ++i)
      for (int j = 0; j < config.hiddenSize; ++j)
        scores[i] += W2[i][j] * hidden[j];
    
    return scores;
  }

  public double[][] getW1() {
    return W1;
  }

  public double[] getb1() {
    return b1;
  }

  public double[][] getW2() {
    return W2;
  }

  public double[][] getE() {
    return E;
  }

  /**
   * Add the two 2d arrays in place of {@code m1}.
   *
   * @throws java.lang.IndexOutOfBoundsException (possibly) If
   *                                             {@code m1} and {@code m2} are not of the same dimensions
   */
  private static void addInPlace(double[][] m1, double[][] m2) {
    for (int i = 0; i < m1.length; i++)
      for (int j = 0; j < m1[0].length; j++)
        m1[i][j] += m2[i][j];
  }

  /**
   * Add the two 1d arrays in place of {@code a1}.
   *
   * @throws java.lang.IndexOutOfBoundsException (Possibly) if
   *                                             {@code a1} and {@code a2} are not of the same dimensions
   */
  private static void addInPlace(double[] a1, double[] a2) {
    for (int i = 0; i < a1.length; i++)
      a1[i] += a2[i];
  }

  /**
   *   Generate the training example globally and decoding with beam search
   * 
   * @param trainSents
   * @param trainTrees
   * @param batchSize
   * @param regParameter
 * @param dMargin 
 * @param nBeam 
   * @param dropProb
   * @return
   */
public Cost computeGlobalCostFunction(List<GlobalExample> globalExamples, int batchSize,
		double regParameter, double dropOutProb, int nBeam, double dMargin) {
	
	   validateTraining();

	   // note that the batch size here is how many sentences other than feature vector!
	    List<GlobalExample> gloExamples = Util.getRandomSubList(globalExamples, batchSize);

	    //get the examples in the global examples
	    List<Example> examples = new ArrayList<Example>();
 	    for(GlobalExample ge : gloExamples)
	    	examples.addAll(ge.getExamples());
	    
	    // Redo precomputations for only those features which are triggered
	    // by examples in this mini-batch.
	    Set<Integer> toPreCompute = getToPreCompute(examples);
	    preCompute(toPreCompute);

	    // Set up parameters for feedforward
	    FeedforwardParams params = new FeedforwardParams(batchSize, dropOutProb);

	    // Zero out saved-embedding gradients
	    gradSaved = new double[preMap.size()][config.hiddenSize];

	    int numChunks = config.trainingThreads;
	    List<Collection<GlobalExample>> chunks = CollectionUtils.partitionIntoFolds(gloExamples, numChunks);

	    // Submit chunks for processing on separate threads
	    for (Collection<GlobalExample> chunk : chunks)
	      jobHandler.put(new Pair<>(chunk, params));
	    jobHandler.join(false);

	    // Join costs from each chunk
	    Cost cost = null;
	    while (jobHandler.peek()) {
	      Cost otherCost = jobHandler.poll();

	      if (cost == null)
	        cost = otherCost;
	      else
	        cost.merge(otherCost);
	    }

	    if (cost == null)
	      return null;

	    // Backpropagate gradients on saved pre-computed values to actual
	    // embeddings
	    cost.backpropSaved(toPreCompute);

	    cost.addL2Regularization(regParameter);

	    return cost;
}

public void setPreMap(Map<Integer, Integer> preMap2) {

	this.preMap = preMap2;
}


}
