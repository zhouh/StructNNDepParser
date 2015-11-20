package nndep;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.PredictedAnswerAnnotation;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.concurrent.MulticoreWrapper;
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import java.util.ArrayList;
import java.util.Arrays;

import static java.util.stream.Collectors.toSet;


/**
 * Neural network classifier which powers a transition-based dependency parser.
 *
 * This classifier is built to accept distributed-representation inputs, and
 * feeds back errors to these input layers as it learns.
 *
 * In order to train a classifier, instantiate this class using the
 * {@link #Classifier(Config, Dataset, double[][], double[][], double[], double[][], java.util.List)}
 * constructor. (The presence of a non-null dataset signals that we wish to
 * train.) After training by alternating calls to
 * {@link #computeCostFunction(int, double, double)} and,
 * {@link #takeAdaGradientStep(edu.stanford.nlp.parser.nndep.Classifier.Cost, double, double)}
 * , be sure to call {@link #finalizeTraining()} in order to allow the
 * classifier to clean up resources used during training.
 *
 * @author Danqi Chen
 * @author Jon Gauthier
 */
public class Classifier {
	// E: numFeatures x embeddingSize
	// W1: hiddenSize x (embeddingSize x numFeatures)
	// b1: hiddenSize
	// W2: numLabels x hiddenSize

	// Weight matrices
	private final double[][] W1, W2, E;
	private final double[][][] labelLayer;
	private final double[] b1;

	// Global gradSaved
	private double[][] gradSaved;

	// Gradient histories
	private double[][] eg2W1, eg2W2, eg2E;
	private double[] eg2b1;
	private double[][][] eg2LabelLayer; // used for hierachical softmax, weight
										// from action to action label,
										// 3 dimension array because we want to
										// control whether all

	/**
	 * Pre-computed hidden layer unit activations. Each double array within this
	 * data is an entire hidden layer. The sub-arrays are indexed somewhat
	 * arbitrarily; in order to find hidden-layer unit activations for a given
	 * feature ID, use {@link #preMap} to find the proper index into this data.
	 */
	private double[][] saved;

	/**
	 * Describes features which should be precomputed. Each entry maps a feature
	 * ID to its destined index in the saved hidden unit activation data (see
	 * {@link #saved}).
	 */
	private Map<Integer, Integer> preMap;

	/**
	 * Initial training state is dependent on how the classifier is initialized.
	 * We use this flag to determine whether calls to
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
	 * Threaded job input: partition of minibatch; current weights + params
	 * Threaded job output: cost value, weight gradients for partition of
	 * minibatch
	 */
	private final MulticoreWrapper<Pair<Collection<GlobalExample>, FeedforwardParams>, Cost> structJobHandler;

	private final MulticoreWrapper<Pair<Collection<Example>, FeedforwardParams>, Cost> greedyJobHandler;

	private final Config config;

	/**
	 * Number of possible dependency relation labels among which this classifier
	 * will choose.
	 */
	private final int numActType;

	private final int numDepLabel;

	private ParsingSystem system;

	private DependencyParser parser = null;

	/**
	 * Instantiate a classifier with previously learned parameters in order to
	 * perform new inference.
	 *
	 * @param config
	 * @param E
	 * @param W1
	 * @param b1
	 * @param W2
	 * @param preComputed
	 */
	public Classifier(Config config, double[][] E, double[][] W1, double[] b1, double[][] W2, double[][][] labelLayer,
			List<Integer> preComputed) {
		this(config, null, E, W1, b1, W2, labelLayer, preComputed);
	}

	/**
	 * Instantiate a classifier with training data and randomly initialized
	 * parameter matrices in order to begin training.
	 *
	 * @param config
	 * @param dataset
	 * @param E
	 * @param W1
	 * @param b1
	 * @param W2
	 * @param preComputed
	 */
	public Classifier(Config config, Dataset dataset, double[][] E, double[][] W1, double[] b1, double[][] W2,
			double[][][] labelLayer, List<Integer> preComputed) {
		this.config = config;
		this.dataset = dataset;

		this.E = E;
		this.W1 = W1;
		this.b1 = b1;
		this.W2 = W2;
		this.labelLayer = labelLayer;

		initGradientHistories();

		numActType = W2.length;
		numDepLabel = labelLayer[0].length;

		preMap = new HashMap<>();
		for (int i = 0; i < preComputed.size(); ++i)
			preMap.put(preComputed.get(i), i);

		isTraining = dataset != null;
		if (isTraining)
			if (config.globalTraining) {
				structJobHandler = new MulticoreWrapper<>(config.trainingThreads, new GlobalCostFunction(), false);
				greedyJobHandler = null;
			} else {
				structJobHandler = null;
				greedyJobHandler = new MulticoreWrapper<>(config.trainingThreads, new GreedyCostFunction(), false);
			}
		else {
			structJobHandler = null;
			greedyJobHandler = null;
		}
	}

	/**
	 * set the parsing system
	 */
	public void setParsingSystem(ParsingSystem system) {
		this.system = system;
	}

	public void setParser(DependencyParser parser) {
		this.parser = parser;
	}

	/**
	 * greedy classifier for training
	 * 
	 * @author zhouh
	 *
	 */
	private class GreedyCostFunction
			implements ThreadsafeProcessor<Pair<Collection<Example>, FeedforwardParams>, Cost> {

		private double[][] gradW1;
		private double[] gradb1;
		private double[][] gradW2;
		private double[][] gradE;
		private double[][][] gradLabelLayer;

		@Override
		public Cost process(Pair<Collection<Example>, FeedforwardParams> input) {
			Collection<Example> examples = input.first();
			FeedforwardParams params = input.second();

			// We can't fix the seed used with ThreadLocalRandom
			// TODO: Is this a serious problem?
			ThreadLocalRandom random = ThreadLocalRandom.current();

			gradW1 = new double[W1.length][W1[0].length];
			gradb1 = new double[b1.length];
			gradW2 = new double[W2.length][W2[0].length];
			gradE = new double[E.length][E[0].length];
			gradLabelLayer = new double[labelLayer.length][labelLayer[0].length][labelLayer[0][0].length];

			double cost = 0.0;
			double correct = 0.0;

			for (Example ex : examples) {
				List<Integer> feature = ex.getFeature();
				List<Integer> actTypeLabel = ex.getactLabel();
				List<Integer> depTypeLabel = ex.getDepLabelLabel();

				double[] actScores = new double[numActType];
				double[] depLabelScores = new double[numDepLabel];
				double[] hidden = new double[config.hiddenSize];
				double[] hidden3 = new double[config.hiddenSize];

				// Run dropout: randomly drop some hidden-layer units. `ls`
				// contains the indices of those units which are still active
				int[] ls = IntStream.range(0, config.hiddenSize)
						.filter(n -> random.nextDouble() > params.getDropOutProb()).toArray();

				int offset = 0;
				for (int j = 0; j < Config.numTokens; ++j) {
					int tok = feature.get(j);
					int index = tok * Config.numTokens + j;

					if (preMap.containsKey(index)) {
						// Unit activations for this input feature value have
						// been
						// precomputed
						int id = preMap.get(index);

						// Only extract activations for those nodes which are
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
				 * begin to hierarchical parsing
				 */
				int optactType = -1;
				int oracleActType = -1;
				for (int i = 0; i < numActType; ++i) {
					if (actTypeLabel.get(i) >= 0) {
						if (actTypeLabel.get(i) == 1)
							oracleActType = i;
						for (int nodeIndex : ls)
							actScores[i] += W2[i][nodeIndex] * hidden3[nodeIndex];

						if (optactType < 0 || actScores[i] > actScores[optactType])
							optactType = i;
					}
				}

				double sum1 = 0.0;
				double sum2 = 0.0;
				double maxScore = actScores[optactType];
				for (int i = 0; i < numActType; ++i) {
					if (actTypeLabel.get(i) >= 0) {
						actScores[i] = Math.exp(actScores[i] - maxScore);
						if (actTypeLabel.get(i) == 1)
							sum1 += actScores[i];
						sum2 += actScores[i];
					}
				}

				/*
				 * softmax on the valid dep labels
				 */
				int optDepType = -1;
				double depTypeSum2 = 0.0;

				if (oracleActType != ParsingSystem.shiftActTypeID) // system.nShift ==
															// 0
				{
					for (int i = 0; i < numDepLabel; ++i) {

						if (depTypeLabel.get(i) >= 0) {
							for (int nodeIndex : ls)
								depLabelScores[i] += labelLayer[oracleActType][i][nodeIndex] * hidden3[nodeIndex]; // change
																													// the
																													// index//////////////////////

							if (optDepType < 0 || depLabelScores[i] > depLabelScores[optDepType])
								optDepType = i;
						}
					}

					double labelMaxScore = depLabelScores[optDepType];
					for (int i = 0; i < numDepLabel; ++i) {
						if (depTypeLabel.get(i) >= 0) {
							depLabelScores[i] = Math.exp(depLabelScores[i] - labelMaxScore);
							depTypeSum2 += depLabelScores[i];
						}
					}
				}

				cost += (Math.log(sum2) - Math.log(sum1)) / params.getBatchSize();

				if (actTypeLabel.get(optactType) == 1)
					correct += +1.0 / params.getBatchSize();

				double[] gradHidden3 = new double[config.hiddenSize];

				for (int i = 0; i < numActType; ++i)
					if (actTypeLabel.get(i) >= 0) {
						double delta = -(actTypeLabel.get(i) - actScores[i] / sum2) / params.getBatchSize();
						for (int nodeIndex : ls) {
							gradW2[i][nodeIndex] += delta * hidden3[nodeIndex];
							gradHidden3[nodeIndex] += delta * W2[i][nodeIndex];
						}
					}

				if (oracleActType != ParsingSystem.shiftActTypeID) {
					for (int i = 0; i < numDepLabel; ++i)
						if (depTypeLabel.get(i) >= 0) {
							double delta = -(depTypeLabel.get(i) - depLabelScores[i] / depTypeSum2)
									/ params.getBatchSize();
							for (int nodeIndex : ls) {
								gradLabelLayer[oracleActType][i][nodeIndex] += delta * hidden3[nodeIndex]; // change
																											// the
																											// index//////////////////////
								gradHidden3[nodeIndex] += delta * labelLayer[oracleActType][i][nodeIndex]; // change
																											// the
																											// index//////////////////////
							}
						}
				}

				double[] gradHidden = new double[config.hiddenSize];
				for (int nodeIndex : ls) {
					gradHidden[nodeIndex] = gradHidden3[nodeIndex] * 3 * hidden[nodeIndex] * hidden[nodeIndex];
					gradb1[nodeIndex] += gradHidden3[nodeIndex];
				}

				offset = 0;
				for (int j = 0; j < Config.numTokens; ++j) {
					int tok = feature.get(j);
					int index = tok * Config.numTokens + j;
					if (preMap.containsKey(index)) {
						int id = preMap.get(index);
						for (int nodeIndex : ls)
							gradSaved[id][nodeIndex] += gradHidden[nodeIndex];
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

			return new Cost(cost, correct, gradW1, gradb1, gradW2, gradLabelLayer, gradE);
		}

		/**
		 * Return a new threadsafe instance.
		 */
		@Override
		public ThreadsafeProcessor<Pair<Collection<Example>, FeedforwardParams>, Cost> newInstance() {
			return new GreedyCostFunction();
		}
	}

	private class GlobalCostFunction
			implements ThreadsafeProcessor<Pair<Collection<GlobalExample>, FeedforwardParams>, Cost> {

		private double[][] gradW1;
		private double[] gradb1;
		private double[][] gradW2;
		private double[][] gradE;
		private double[][][] gradLabelLayer;

		@Override
		public Cost process(Pair<Collection<GlobalExample>, FeedforwardParams> input) {
			Collection<GlobalExample> examples = input.first();
			FeedforwardParams params = input.second();

			gradW1 = new double[W1.length][W1[0].length];
			gradb1 = new double[b1.length];
			gradW2 = new double[W2.length][W2[0].length];
			gradE = new double[E.length][E[0].length];
			gradLabelLayer = new double[labelLayer.length][labelLayer[0].length][labelLayer[0][0].length];

			double cost = 0.0;
			double correct = 0.0;

			for (GlobalExample ex : examples) {

				Triple<Double, Boolean, ActTypeBeam> predict = multiBeamDecoding(params, true, null, ex);

				correct += predict.first;

				/*
				 * Begin to train!
				 */
				int nActTypeBeamSize = config.nHierarchyActTypeBeamSize;
				int nDepTypeBeamSize = config.nHierarchiyDepTypeBeamSize;
				ActTypeBeam actTypeBeam = predict.third;
				HierarchicalDepState predictState = actTypeBeam.getBestState();
				// softmax the whole beam candidates!

				if (!config.bAggressiveUpdate && !predict.second) {
					if (predictState.bGold)
						continue; // skip update if predict right!
				}

				double[][] actTypeUpdatePara = new double[nActTypeBeamSize][nDepTypeBeamSize];
				double[][] depTypeUpdatePara = new double[nActTypeBeamSize][nDepTypeBeamSize];
				
				double sumActTypeScores = 0;
				double sumDepTypeScores = 0;
				double bestATScore = actTypeBeam.bestActTypeScore;
				double bestDTScore = actTypeBeam.bestDepTypeScore;
				
				int dtI = 0;
				for (DepTypeBeam dtBeam : actTypeBeam) {
					int sI = 0;
					for (HierarchicalDepState state : dtBeam.bucket) {
						
						actTypeUpdatePara[dtI][sI] = Math.exp(state.actTypeScore - bestATScore);
						sumActTypeScores += actTypeUpdatePara[dtI][sI];
						
						depTypeUpdatePara[dtI][sI] = Math.exp(state.depTypeScore - bestDTScore);
						sumDepTypeScores += depTypeUpdatePara[dtI][sI];
						
						sI++;
					}
					
					dtI++;
				}
				
				double[][] actTypeUpdateParaUnNorm = new double[nActTypeBeamSize][nDepTypeBeamSize];
				double[][] depTypeUpdateParaUnNorm = new double[nActTypeBeamSize][nDepTypeBeamSize];
				actTypeUpdateParaUnNorm = actTypeUpdatePara.clone();
				depTypeUpdateParaUnNorm = depTypeUpdatePara.clone();
				
				// set the gradients of each beam candidate
				// here we use the Roundoff error trick in softmax, please
				// check the book of "Neural Network, Tricks and Trade" for the detail
				dtI = 0;
				for (DepTypeBeam dtBeam : actTypeBeam) {
					int sI = 0;
					for (HierarchicalDepState state : dtBeam.bucket) {
						int t = state.bGold ? 1 : 0;
						actTypeUpdatePara[dtI][sI] = actTypeUpdatePara[dtI][sI] / sumActTypeScores;
						depTypeUpdatePara[dtI][sI] = depTypeUpdatePara[dtI][sI] / sumDepTypeScores;
						
						if(actTypeUpdatePara[dtI][sI] <= 0.5)
							actTypeUpdatePara[dtI][sI] = actTypeUpdatePara[dtI][sI] - t;
						else
							actTypeUpdatePara[dtI][sI] = (1 - t) - 
							(sumActTypeScores - actTypeUpdateParaUnNorm[dtI][sI]) / sumActTypeScores;
						
						if(depTypeUpdatePara[dtI][sI] <= 0.5)
							depTypeUpdatePara[dtI][sI] = depTypeUpdatePara[dtI][sI] - t;
						else
							depTypeUpdatePara[dtI][sI] = (1 - t) - 
							(sumDepTypeScores - depTypeUpdateParaUnNorm[dtI][sI]) / sumDepTypeScores;
						
					}
				}

				/*
				 * training procedure
				 */
				dtI = 0;
				for (DepTypeBeam dtBeam : actTypeBeam) {
					int sI = 0;
					for (HierarchicalDepState state : dtBeam.bucket) {
						
						while(state.lastState != null){
							trainOneState(params, state, 
									actTypeUpdatePara[dtI][sI], depTypeUpdatePara[dtI][sI]);
							state = state.lastState;
						}
						
					}
				}

			} // end foreach examples

			return new Cost(cost, correct, gradW1, gradb1, gradW2, gradLabelLayer, gradE);
		}

		/**
		 * back propagation update for a given state
		 * 
		 * @param params
		 * @param state
		 * @param actTypeUpdatePara
		 * @param depTypeUpdatePara
		 */
		private void trainOneState(FeedforwardParams params, HierarchicalDepState state, 
				double actTypeUpdatePara, double depTypeUpdatePara) {
			
			int[] features = state.lastState.featureArray;
			double[] hidden = state.hiddenLayer.hidden;
			double[] hidden3 = state.hiddenLayer.hidden3;
			double[] gradHidden3 = new double[config.hiddenSize];
			double[] gradHidden = new double[config.hiddenSize];
			int[] dropOut = state.lastState.hiddenLayer.dropOut;
			int optActType = state.actType;
			int optDepType = state.depType;
			
			/*
			 * update the actType layer
			 */
			double actTypeDelta = actTypeUpdatePara / params.getBatchSize();
			for (int nodeIndex : dropOut) {
				gradW2[optActType][nodeIndex] += actTypeDelta * hidden3[nodeIndex];
				gradHidden3[nodeIndex] += actTypeDelta * W2[optActType][nodeIndex];
			}
			
			/*
			 * update the depType layer
			 */
			if(optActType != ParsingSystem.shiftActTypeID){
				double depTypeDelta = depTypeUpdatePara / params.getBatchSize();
				for (int nodeIndex : dropOut) {
					gradLabelLayer[optActType][optDepType][nodeIndex] += depTypeDelta * hidden3[nodeIndex];
					gradHidden3[nodeIndex] += actTypeDelta * gradLabelLayer[optActType][optDepType][nodeIndex];
				}
			}
			
			
			for (int nodeIndex : dropOut) {
				gradHidden[nodeIndex] = gradHidden3[nodeIndex] * 3 * hidden[nodeIndex] * hidden[nodeIndex];
				gradb1[nodeIndex] += gradHidden3[nodeIndex];
			}

			int offset = 0;
			for (int j = 0; j < Config.numTokens; ++j) {
				int tok = features[j];
				int index = tok * Config.numTokens + j;
				if (preMap.containsKey(index)) {
					int id = preMap.get(index);
					for (int nodeIndex : dropOut) {
						gradSaved[id][nodeIndex] += gradHidden[nodeIndex];
					}
				} else {
					for (int nodeIndex : dropOut) {
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
			return new GlobalCostFunction();
		}
	}

	/**
	 * Describes the parameters for a particular invocation of a cost function.
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
	 * Describes the result of feedforward + backpropagation through the neural
	 * network for the batch provided to a `CostFunction.`
	 * <p>
	 * The members of this class represent weight deltas computed by
	 * backpropagation.
	 *
	 * @see Classifier.GlobalCostFunction
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
		private final double[][][] gradLabelLayer;

		private Cost(double cost, double percentCorrect, double[][] gradW1, double[] gradb1, double[][] gradW2,
				double[][][] gradLabelLayer, double[][] gradE) {
			this.cost = cost;
			this.percentCorrect = percentCorrect;

			this.gradW1 = gradW1;
			this.gradb1 = gradb1;
			this.gradW2 = gradW2;
			this.gradLabelLayer = gradLabelLayer;
			this.gradE = gradE;
		}

		/**
		 * Merge the given {@code Cost} data with the data in this instance.
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
			addInPlace(gradLabelLayer, otherCost.gradLabelLayer);
		}

		/**
		 * Backpropagate gradient values from gradSaved into the gradients for
		 * the E vectors that generated them.
		 *
		 * @param featuresSeen
		 *            Feature IDs observed during training for which gradSaved
		 *            values need to be backprop'd into gradE
		 */
		private void backpropSaved(Set<Integer> featuresSeen) {
			for (int x : featuresSeen) {
				int mapX = preMap.get(x);
				int tok = x / Config.numTokens;
				int offset = (x % Config.numTokens) * config.embeddingSize;
				for (int j = 0; j < config.hiddenSize; ++j) {
					double delta = gradSaved[mapX][j];
					for (int k = 0; k < config.embeddingSize; ++k) {
						gradW1[j][offset + k] += delta * E[tok][k];

						if (Double.isInfinite(gradW1[j][offset + k]) || Double.isNaN(gradW1[j][offset + k])) {
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

			for (int i = 0; i < labelLayer.length; ++i) {
				if (i == ParsingSystem.shiftActTypeID)
					continue;
				for (int j = 0; j < labelLayer[0].length; ++j)
					for (int k = 0; k < labelLayer[0][0].length; ++k) {
						cost += regularizationWeight * labelLayer[i][j][k] * labelLayer[i][j][k] / 2.0;
						gradLabelLayer[i][j][k] += regularizationWeight * labelLayer[i][j][k];
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

		public double[][][] getGradLabelLayer() {
			return gradLabelLayer;
		}

	}

	/**
	 * Determine the feature IDs which need to be pre-computed for training with
	 * these examples.
	 */
	private Set<Integer> getToPreCompute(List<Example> examples) {
		Set<Integer> featureIDs = new HashSet<>();
		for (Example ex : examples) {
			List<Integer> feature = ex.getFeature();

			for (int j = 0; j < Config.numTokens; j++) {
				int tok = feature.get(j);
				int index = tok * Config.numTokens + j;
				if (preMap.containsKey(index))
					featureIDs.add(index);
			}
		}

		double percentagePreComputed = featureIDs.size() / (float) config.numPreComputed;
		System.err.printf("Percent actually necessary to pre-compute: %f%%%n", percentagePreComputed * 100);

		return featureIDs;
	}

	/**
	 * Determine the total cost on the dataset associated with this classifier
	 * using the current learned parameters. This cost is evaluated using
	 * mini-batch adaptive gradient descent.
	 *
	 * This method launches multiple threads, each of which evaluates training
	 * cost on a partition of the mini-batch.
	 *
	 * @param batchSize
	 * @param regParameter
	 *            Regularization parameter (lambda)
	 * @param dropOutProb
	 *            Drop-out probability. Hidden-layer units in the neural network
	 *            will be randomly turned off while training a particular
	 *            example with this probability.
	 * @return A {@link edu.stanford.nlp.parser.nndep.Classifier.Cost} object
	 *         which describes the total cost of the given weights, and includes
	 *         gradients to be used for further training
	 */
	public Cost computeGreedyCostFunction(int batchSize, double regParameter, double dropOutProb) {
		validateTraining();

		List<Example> examples = Util.getRandomSubList(dataset.examples, batchSize);

		// Redo precomputations for only those features which are triggered
		// by examples in this mini-batch.
		Set<Integer> toPreCompute = getToPreCompute(examples);
		preCompute(toPreCompute);

		// Set up parameters for feedforward
		FeedforwardParams params = new FeedforwardParams(batchSize, dropOutProb);

		// Zero out saved-embedding gradients
		gradSaved = new double[preMap.size()][config.hiddenSize];

		int numChunks = config.trainingThreads;
		List<Collection<Example>> chunks = CollectionUtils.partitionIntoFolds(examples, numChunks);

		// Submit chunks for processing on separate threads
		for (Collection<Example> chunk : chunks)
			greedyJobHandler.put(new Pair<>(chunk, params));
		greedyJobHandler.join(false);

		// Join costs from each chunk
		Cost cost = null;
		while (greedyJobHandler.peek()) {
			Cost otherCost = greedyJobHandler.poll();

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

	/**
	 * Update classifier weights using the given training cost information.
	 *
	 * @param cost
	 *            Cost information as returned by
	 *            {@link #computeCostFunction(int, double, double)}.
	 * @param adaAlpha
	 *            Global AdaGrad learning rate
	 * @param adaEps
	 *            Epsilon value for numerical stability in AdaGrad's division
	 */
	public void takeAdaGradientStep(Cost cost, double adaAlpha, double adaEps) {
		validateTraining();

		double[][] gradW1 = cost.getGradW1(), gradW2 = cost.getGradW2(), gradE = cost.getGradE();
		double[] gradb1 = cost.getGradb1();
		double[][][] gradLabelLayer = cost.getGradLabelLayer();

		for (int i = 0; i < W1.length; ++i) {
			for (int j = 0; j < W1[i].length; ++j) {
				eg2W1[i][j] += gradW1[i][j] * gradW1[i][j];
				if (Double.isInfinite(gradW1[i][j]) || Double.isNaN(gradW1[i][j])) {
					throw new RuntimeException("Infinite or Nan gradW1!");
					// System.err.println(gradW1[i][j]) );
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

		for (int i = 0; i < labelLayer.length; ++i) {
			if (i == ParsingSystem.shiftActTypeID)
				continue;
			for (int j = 0; j < labelLayer[0].length; ++j)
				for (int k = 0; k < labelLayer[0][0].length; ++k) {
					eg2LabelLayer[i][j][k] += gradLabelLayer[i][j][k] * gradLabelLayer[i][j][k];
					labelLayer[i][j][k] -= adaAlpha * gradLabelLayer[i][j][k]
							/ Math.sqrt(eg2LabelLayer[i][j][k] + adaEps);
				}
			// break;
		}

	}

	private void initGradientHistories() {
		eg2E = new double[E.length][E[0].length];
		eg2W1 = new double[W1.length][W1[0].length];
		eg2b1 = new double[b1.length];
		eg2W2 = new double[W2.length][W2[0].length];
		eg2LabelLayer = new double[labelLayer.length][labelLayer[0].length][labelLayer[0][0].length];
	}

	/**
	 * Clear all gradient histories used for AdaGrad training.
	 *
	 * @throws java.lang.IllegalStateException
	 *             If not training
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
		if (config.globalTraining)
			structJobHandler.join(true);
		else
			greedyJobHandler.join(true);

		isTraining = false;
	}

	/**
	 * @see #preCompute(java.util.Set)
	 */
	public void preCompute() {
		// If no features are specified, pre-compute all of them (which fit
		// into a `saved` array of size `config.numPreComputed`)
		Set<Integer> keys = preMap.entrySet().stream().filter(e -> e.getValue() < config.numPreComputed)
				.map(Map.Entry::getKey).collect(toSet());
		preCompute(keys);
	}

	/**
	 * Pre-compute hidden layer activations for some set of possible feature
	 * inputs.
	 *
	 * @param toPreCompute
	 *            Set of feature IDs for which hidden layer activations should
	 *            be precomputed
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
			int tok = x / Config.numTokens;
			int pos = x % Config.numTokens;
			for (int j = 0; j < config.hiddenSize; ++j)
				for (int k = 0; k < config.embeddingSize; ++k)
					saved[mapX][j] += W1[j][pos * config.embeddingSize + k] * E[tok][k];
		}
		System.err.println("PreComputed " + toPreCompute.size() + ", Elapsed Time: "
				+ (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");
	}
	
	public void DecodingInTest(CoreMap sentence){
		
	}

	/**
	 *  the multi-beam decoding function
	 *  
	 */
	public Triple<Double, Boolean, ActTypeBeam> multiBeamDecoding( FeedforwardParams params, boolean bTrain, 
			CoreMap s, GlobalExample ex ){
		
		CoreMap sentence = bTrain ? ex.sent : s;
		List<Pair<Integer, Integer>> oracles = ex.oracles;

		/*
		 * Begin to decode!
		 */
		int nActTypeBeamSize = config.nHierarchyActTypeBeamSize;
		int nDepTypeBeamSize = config.nHierarchiyDepTypeBeamSize;

		int nSentSize = sentence.get(CoreAnnotations.TokensAnnotation.class).size();
		int nRound = nSentSize * 2;
		boolean updateEarly = false; // whether this example execute
									 // early update

		Configuration c = system.initialConfiguration(sentence);

		// only store the best beam candidates in decoding!
		HierarchicalDepState initialState = new HierarchicalDepState(c, -1, -1, 0.0, 0.0, 0.0, null, true);
		DepTypeBeam depTypeBeam = new DepTypeBeam(nDepTypeBeamSize);
		ActTypeBeam actTypeBeam = new ActTypeBeam(nActTypeBeamSize);
		ActTypeBeam actTypeBeamAfterExpand = new ActTypeBeam(nActTypeBeamSize);
		actTypeBeamAfterExpand.clearAll();
		depTypeBeam.insert(initialState);
		actTypeBeam.insert(depTypeBeam);

		// begin to do nRound-th action
		int ri;
		for (ri = 0; ri < nRound; ri++) {

			Pair<Integer, Integer> goldAction = null;
			if(bTrain)
				goldAction = oracles.get(ri);

			/*
			 * begin to expand hierarchical multi-beam decoding!
			 * 
			 * first to expand the acttype beam and prune, then expand
			 * every state in the bucket of deptype beam, and prune.
			 * 
			 */

			// first, expand the action type for every state
			for (DepTypeBeam dtBeam : actTypeBeam) { // for every bucket
														// in beam

				DepTypeBeam[] expandedDepTypeBeams = new DepTypeBeam[ParsingSystem.nActTypeNum];

				for (int i = 0; i < expandedDepTypeBeams.length; i++)
					expandedDepTypeBeams[i] = new DepTypeBeam(nDepTypeBeamSize);

				for (HierarchicalDepState state : dtBeam) { // for every
															// state

					// get score for every state
					int[] featureArray = parser.getFeatureArray(state.c);
					HiddenLayer hiddenLayer = getHidden(bTrain, featureArray, state.c);
					int[] actTypeLabel = system.getValidActType(state.c);
					double[] actLayerScore = getActTypeLayer(bTrain, hiddenLayer, actTypeLabel);

					if(bTrain)	// only in training, we need these paras for updating
						state.setParas(hiddenLayer, actTypeLabel, featureArray);

					/*
					 * for every action type
					 */
					for (int actTypeI = 0; actTypeI < ParsingSystem.nActTypeNum; actTypeI++) { 

						if (actTypeLabel[actTypeI] < 0)
							continue; // skip the invalid action

						/*
						 *  the score of next state is the sum of current
						 *  act type scores and dep type scores
						 */
						HierarchicalDepState newState;
						if(bTrain == true)
							newState = new HierarchicalDepState(
									state.c, 
									actTypeI,
									-1,
									state.score, 
									state.actTypeScore + actLayerScore[actTypeI], 
									state.depTypeScore,
									state, 
									state.bGold && actTypeI == goldAction.first);
						else
							newState = new HierarchicalDepState(
									state.c, 
									actTypeI,
									-1,
									state.score, 
									state.actTypeScore + actLayerScore[actTypeI], 
									state.depTypeScore,
									state, 
									false);
						
						expandedDepTypeBeams[actTypeI].insertBucket(newState);
					}
				}

				// add expanded depType beam into new act type beam
				for (DepTypeBeam db : expandedDepTypeBeams)
					actTypeBeamAfterExpand.insert(db);
			}

			// second, expand all the states in the pruned act type beam
			// with different dep type
			int nDepTypeNum = system.labels.size();
			for (DepTypeBeam dtBeam : actTypeBeamAfterExpand) {

				boolean bLastActShift = false;
				for (HierarchicalDepState state : dtBeam.bucket) {

					int givenActType = state.actType;
					if (givenActType == ParsingSystem.shiftActTypeID) { // if last action is
																        // shift, then skip
						bLastActShift = true;
						break;
					}

					// get dep type layer for every state
					int[] depTypeLabel = system.getValidLabelGivenActType(state.c, givenActType);
					double[] detTypeScores = getDepTypeLayer(bTrain, givenActType, state.hiddenLayer,
							depTypeLabel);

					for (int depTypeI = 0; depTypeI < nDepTypeNum; depTypeI++) {
						
						if(depTypeLabel[depTypeI] < 0) continue; //invalid? skip
						
						HierarchicalDepState newState;
						if(bTrain == true)
							newState = state.getNewStateForExpandDepType(depTypeI,
									state.depTypeScore + detTypeScores[depTypeI],
									depTypeI == goldAction.second && state.bGold, 
									depTypeLabel);
						else
							newState = state.getNewStateForExpandDepType(depTypeI,
									state.depTypeScore + detTypeScores[depTypeI],
									false, 
									depTypeLabel);

						dtBeam.insert(newState);
					}
				}

				if (bLastActShift)
					continue;
			}

			// early update
			if (bTrain && config.earlyUpdate) {
				if (!actTypeBeamAfterExpand.containGold()) {
					updateEarly = true;
					break;
				}
			}

			// apply these states, lazy expand
			for (DepTypeBeam dtBeam : actTypeBeamAfterExpand)
				for (HierarchicalDepState state : dtBeam)
					state.StateApply(system);

			actTypeBeam.clearAll();
			actTypeBeam = actTypeBeamAfterExpand;

		} // end nRound

		int correct = (ri / nSentSize / params.batchSize) / 2;

		return new Triple(correct, updateEarly, actTypeBeam);
		
	}
	

	public Pair<Integer, Integer> computeHierarchicalScore(int[] feature, Configuration c) {

		int[] actTypeLabel = system.getValidActType(c);

		double[] hidden = new double[config.hiddenSize];
		int offset = 0;
		for (int j = 0; j < feature.length; ++j) {
			int tok = feature[j];
			int index = tok * Config.numTokens + j;

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
			hidden[i] = hidden[i] * hidden[i] * hidden[i]; // cube nonlinearity
		}

		double[] actTypeScores = new double[numActType];
		int optActType = -1;
		double optActTypeScore = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < numActType; ++i) {

			if (actTypeLabel[i] == -1)
				continue;
			for (int j = 0; j < config.hiddenSize; ++j)
				actTypeScores[i] += W2[i][j] * hidden[j];

			if (actTypeScores[i] >= optActTypeScore) {
				optActTypeScore = actTypeScores[i];
				optActType = i;
			}
		}

		int[] depTypeLabel = system.getValidLabelGivenActType(c, optActType);
		/*
		 * for dep labels
		 */
		double[] depTypeScores = new double[numDepLabel];
		int optDepType = -1;

		if (optActType != ParsingSystem.shiftActTypeID) {
			double optDepTypeScore = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < numDepLabel; ++i) {

				if (depTypeLabel[i] == -1)
					continue;
				for (int j = 0; j < config.hiddenSize; ++j)
					depTypeScores[i] += labelLayer[optActType][i][j] * hidden[j]; // change
																					// the
																					// index//////////////////////

				if (depTypeScores[i] >= optDepTypeScore) {
					optDepTypeScore = depTypeScores[i];
					optDepType = i;
				}
			}
		}

		return new Pair<Integer, Integer>(optActType, optDepType);
	}

	public Pair<Integer, Double> getOptActType(int[] feature, Configuration c) {

		int[] actTypeLabel = system.getValidActType(c);

		double[] hidden = new double[config.hiddenSize];
		int offset = 0;
		for (int j = 0; j < feature.length; ++j) {
			int tok = feature[j];
			int index = tok * Config.numTokens + j;

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
			hidden[i] = hidden[i] * hidden[i] * hidden[i]; // cube nonlinearity
		}

		double[] actTypeScores = new double[numActType];
		int optActType = -1;
		double optActTypeScore = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < numActType; ++i) {

			if (actTypeLabel[i] == -1)
				continue;
			for (int j = 0; j < config.hiddenSize; ++j)
				actTypeScores[i] += W2[i][j] * hidden[j];

			if (actTypeScores[i] >= optActTypeScore) {
				optActTypeScore = actTypeScores[i];
				optActType = i;
			}
		}

		double sum2 = 0.0;
		double maxScore = actTypeScores[optActType];
		for (int i = 0; i < numActType; ++i) {
			if (actTypeLabel[i] >= 0) {
				actTypeScores[i] = Math.exp(actTypeScores[i] - maxScore);
				sum2 += actTypeScores[i];
			}
		}

		for (int i = 0; i < numActType; ++i) {
			if (actTypeLabel[i] >= 0) {
				actTypeScores[i] = actTypeScores[i] / sum2;
			}
		}

		return new Pair<>(optActType, actTypeScores[optActType]);
	}

	public int getOptDepTypeGivenActType(int[] feature, Configuration c, int optActType) {
		double[] hidden = new double[config.hiddenSize];
		int offset = 0;
		for (int j = 0; j < feature.length; ++j) {
			int tok = feature[j];
			int index = tok * Config.numTokens + j;

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
			hidden[i] = hidden[i] * hidden[i] * hidden[i]; // cube nonlinearity
		}

		int[] depTypeLabel = system.getValidLabelGivenActType(c, optActType);
		/*
		 * for dep labels
		 */
		double[] depTypeScores = new double[numDepLabel];
		int optDepType = -1;

		if (optActType != ParsingSystem.shiftActTypeID) {
			double optDepTypeScore = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < numDepLabel; ++i) {

				if (depTypeLabel[i] == -1)
					continue;
				for (int j = 0; j < config.hiddenSize; ++j)
					depTypeScores[i] += labelLayer[optActType][i][j] * hidden[j]; // change
																					// the
																					// index//////////////////////

				if (depTypeScores[i] >= optDepTypeScore) {
					optDepTypeScore = depTypeScores[i];
					optDepType = i;
				}
			}
		}

		return optDepType;
	}

	public HiddenLayer getHidden(boolean bTrain, int[] feature, Configuration c) {

		double[] hidden = new double[config.hiddenSize];
		double[] hidden3 = new double[config.hiddenSize];

		ThreadLocalRandom random = ThreadLocalRandom.current();

		// Run dropout: randomly drop some hidden-layer units. `ls`
		// contains the indices of those units which are still active
		int[] ls = null;
		if(bTrain)
			ls = IntStream.range(0, config.hiddenSize).filter(n -> random.nextDouble() > config.dropProb).toArray();
		
		int offset = 0;
		for (int j = 0; j < Config.numTokens; ++j) {
			int tok = feature[j];
			int index = tok * Config.numTokens + j;

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

		return new HiddenLayer(hidden, hidden3, ls);

	}

	public double[] getActTypeLayer(boolean bTrain, HiddenLayer hiddenLayer, int[] actTypeLabel) {

		double[] actTypeScore = new double[ParsingSystem.nActTypeNum];

		for (int i = 0; i < numActType; ++i)
			if (actTypeLabel[i] >= 0)
				if(bTrain)
					for (int nodeIndex : hiddenLayer.dropOut)
						actTypeScore[i] += W2[i][nodeIndex] * hiddenLayer.hidden3[nodeIndex];
				else
					for(int j = 0; j < config.hiddenSize; j++)
						actTypeScore[i] += W2[i][j] * hiddenLayer.hidden3[j];
		
		return actTypeScore;
	}

	public double[] getDepTypeLayer(boolean bTrain, int givenActType, HiddenLayer hiddenLayer, int[] depTypeLabel) {

		double[] depLabelScores = new double[system.labels.size()];

		for (int i = 0; i < system.labels.size(); ++i)
			if (depTypeLabel[i] >= 0)
				if(bTrain)
					for (int nodeIndex : hiddenLayer.dropOut)
						depLabelScores[i] += labelLayer[givenActType][i][nodeIndex] * hiddenLayer.hidden3[nodeIndex]; // change
				else
					for(int j = 0; j < config.hiddenSize; j++)																								// the
						depLabelScores[i] += labelLayer[givenActType][i][j] * hiddenLayer.hidden3[j];																						// index//////////////////////

		return depLabelScores;
	}

	// double[] computeScores(int[] feature) {
	// return computeScores(feature, preMap);
	// }

	/**
	 * ComputerScoresInTraining
	 * 
	 * In training, we need to decode with dropout, so the drop out result is
	 * computed before computer scores in this function! The drop out array will
	 * be used in update parameters!
	 * 
	 */
	// private double[] computeScoresInTraining(int[] feature, Map<Integer,
	// Integer> preMap,
	// int[] dropOutArray){
	//
	// int[] ls = dropOutArray;
	// double[] hidden = new double[config.hiddenSize];
	// int offset = 0;
	// for (int j = 0; j < config.numTokens; ++j) {
	// int tok = feature[j];
	// int index = tok * config.numTokens + j;
	//
	// if (preMap.containsKey(index)) {
	// int id = preMap.get(index);
	//
	// // Only extract activations for those nodes which are still
	// // activated (`ls`)
	// for (int nodeIndex : ls)
	// hidden[nodeIndex] += saved[id][nodeIndex];
	// } else {
	// for (int nodeIndex : ls) {
	// for (int k = 0; k < config.embeddingSize; ++k)
	// hidden[nodeIndex] += W1[nodeIndex][offset + k] * E[tok][k];
	// }
	// }
	// offset += config.embeddingSize;
	// }
	//
	// // Add bias term and apply activation function
	// for (int nodeIndex : ls) {
	// hidden[nodeIndex] += b1[nodeIndex];
	// hidden[nodeIndex] = hidden[nodeIndex] * hidden[nodeIndex] *
	// hidden[nodeIndex]; // cube nonlinearity
	// }
	//
	// // Feed forward to softmax layer (no activation yet)
	// double[] scores = new double[numLabels];
	// for (int i = 0; i < numLabels; ++i) {
	// for (int nodeIndex : ls)
	// scores[i] += W2[i][nodeIndex] * hidden[nodeIndex];
	// }
	//
	// return scores;
	// }

	/**
	 * Feed a feature vector forward through the network. Returns the values of
	 * the output layer.
	 */
	// private double[] computeScores(int[] feature, Map<Integer, Integer>
	// preMap) {
	// double[] hidden = new double[config.hiddenSize];
	// int offset = 0;
	// for (int j = 0; j < feature.length; ++j) {
	// int tok = feature[j];
	// int index = tok * config.numTokens + j;
	//
	// if (preMap.containsKey(index)) {
	// int id = preMap.get(index);
	// for (int i = 0; i < config.hiddenSize; ++i)
	// hidden[i] += saved[id][i];
	// } else {
	// for (int i = 0; i < config.hiddenSize; ++i)
	// for (int k = 0; k < config.embeddingSize; ++k)
	// hidden[i] += W1[i][offset + k] * E[tok][k];
	// }
	// offset += config.embeddingSize;
	// }
	//
	// for (int i = 0; i < config.hiddenSize; ++i) {
	// hidden[i] += b1[i];
	// hidden[i] = hidden[i] * hidden[i] * hidden[i]; // cube nonlinearity
	// }
	//
	// double[] scores = new double[numLabels];
	// for (int i = 0; i < numLabels; ++i)
	// for (int j = 0; j < config.hiddenSize; ++j)
	// scores[i] += W2[i][j] * hidden[j];
	//
	// return scores;
	// }

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

	public double[][][] getLabelLayer() {

		return labelLayer;
	}

	/**
	 * Add the two 2d arrays in place of {@code m1}.
	 *
	 * @throws java.lang.IndexOutOfBoundsException
	 *             (possibly) If {@code m1} and {@code m2} are not of the same
	 *             dimensions
	 */
	private static void addInPlace(double[][] m1, double[][] m2) {
		for (int i = 0; i < m1.length; i++)
			for (int j = 0; j < m1[0].length; j++)
				m1[i][j] += m2[i][j];
	}

	/**
	 * Add the two 1d arrays in place of {@code a1}.
	 *
	 * @throws java.lang.IndexOutOfBoundsException
	 *             (Possibly) if {@code a1} and {@code a2} are not of the same
	 *             dimensions
	 */
	private static void addInPlace(double[] a1, double[] a2) {
		for (int i = 0; i < a1.length; i++)
			a1[i] += a2[i];
	}

	private static void addInPlace(double[][][] m1, double[][][] m2) {

		for (int i = 0; i < m1.length; i++) {
			if (i == ParsingSystem.shiftActTypeID)
				continue;
			for (int j = 0; j < m1[0].length; j++)
				for (int k = 0; k > m1[0][0].length; k++)
					m1[i][j][k] += m2[i][j][k];

			// break;
		}
	}

	/**
	 * Generate the training example globally and decoding with beam search
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
	public Cost computeGlobalCostFunction(List<GlobalExample> globalExamples, int batchSize, double regParameter,
			double dropOutProb, int nBeam, double dMargin) {

		validateTraining();

		// note that the batch size here is how many sentences other than
		// feature vector!
		List<GlobalExample> gloExamples = Util.getRandomSubList(globalExamples, batchSize);

		// get the examples in the global examples
		List<Example> examples = new ArrayList<Example>();
		for (GlobalExample ge : gloExamples)
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
			structJobHandler.put(new Pair<>(chunk, params));
		structJobHandler.join(false);

		// Join costs from each chunk
		Cost cost = null;
		while (structJobHandler.peek()) {
			Cost otherCost = structJobHandler.poll();

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
