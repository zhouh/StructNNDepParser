package nndep;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasTag;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeGraphNode;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.Triple;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;

import static java.util.stream.Collectors.toList;

/**
 * This class defines a transition-based dependency parser which makes
 * use of a classifier powered by a neural network. The neural network
 * accepts distributed representation inputs: dense, continuous
 * representations of words, their part of speech tags, and the labels
 * which connect words in a partial dependency parse.
 *
 * <p>
 * This is an implementation of the method described in
 *
 * <blockquote>
 *   Danqi Chen and Christopher Manning. A Fast and Accurate Dependency
 *   Parser Using Neural Networks. In EMNLP 2014.
 * </blockquote>
 *
 * <p>
 * New models can be trained from the command line; see {@link #main}
 * for details on training options. This parser will also output
 * CoNLL-X format predictions; again see {@link #main} for available
 * options.
 *
 * <p>
 * This parser can also be used programmatically. The easiest way to
 * prepare the parser with a pre-trained model is to call
 * {@link #loadFromModelFile(String)}. Then call
 * {@link #predict(edu.stanford.nlp.util.CoreMap)} on the returned
 * parser instance in order to get new parses.
 *
 * @author Danqi Chen (danqi@cs.stanford.edu)
 * @author Jon Gauthier
 */
public class DependencyParser {
  public static final String DEFAULT_MODEL = "edu/stanford/nlp/models/parser/nndep/PTB_Stanford_params.txt.gz";

  /**
   * Words, parts of speech, and dependency relation labels which were
   * observed in our corpus / stored in the model
   *
   * @see #genDictionaries(java.util.List, java.util.List)
   */
  private List<String> knownWords, knownPos, knownLabels;
 

  /**
   * Mapping from word / POS / dependency relation label to integer ID
   */
  private Map<String, Integer> wordIDs, posIDs, labelIDs;

  private List<Integer> preComputed;
  
  public List<GlobalExample> globalExamples;

  /**
   * Given a particular parser configuration, this classifier will
   * predict the best transition to make next.
   *
   * The {@link edu.stanford.nlp.parser.nndep.Classifier} class
   * handles both training and inference.
   */
  private Classifier classifier;

  private ParsingSystem system;

  private Map<String, Integer> embedID;
  private double[][] embeddings;

  private final Config config;

  /**
   * Language used to generate
   * {@link edu.stanford.nlp.trees.GrammaticalRelation} instances.
   */
  private final GrammaticalRelation.Language language;

  DependencyParser() {
    this(new Properties());
  }

  public DependencyParser(Properties properties) {
    config = new Config(properties);

    // Convert Languages.Language instance to
    // GrammaticalLanguage.Language
    switch (config.language) {
      case English:
        language = GrammaticalRelation.Language.English;
        break;
      case Chinese:
        language = GrammaticalRelation.Language.Chinese;
        break;
      default:
        language = GrammaticalRelation.Language.Any;
        break;
    }
  }

  /**
   * Get an integer ID for the given word. This ID can be used to index
   * into the embeddings {@link #embeddings}.
   *
   * @return An ID for the given word, or an ID referring to a generic
   *         "unknown" word if the word is unknown
   */
  public int getWordID(String s) {
      return wordIDs.containsKey(s) ? wordIDs.get(s) : wordIDs.get(Config.UNKNOWN);
  }

  public int getPosID(String s) {
      return posIDs.containsKey(s) ? posIDs.get(s) : posIDs.get(Config.UNKNOWN);
  }

  public int getLabelID(String s) {
    return labelIDs.get(s);
  }

  public List<Integer> getFeatures(Configuration c) {
    // Presize the arrays for very slight speed gain. Hardcoded, but so is the current feature list.
    List<Integer> fWord = new ArrayList<Integer>(18);
    List<Integer> fPos = new ArrayList<Integer>(18);
    List<Integer> fLabel = new ArrayList<Integer>(12);
    for (int j = 2; j >= 0; --j) {
      int index = c.getStack(j);
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
    }
    for (int j = 0; j <= 2; ++j) {
      int index = c.getBuffer(j);
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
    }
    for (int j = 0; j <= 1; ++j) {
      int k = c.getStack(j);
      int index = c.getLeftChild(k);
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
      fLabel.add(getLabelID(c.getLabel(index)));

      index = c.getRightChild(k);
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
      fLabel.add(getLabelID(c.getLabel(index)));

      index = c.getLeftChild(k, 2);
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
      fLabel.add(getLabelID(c.getLabel(index)));

      index = c.getRightChild(k, 2);
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
      fLabel.add(getLabelID(c.getLabel(index)));

      index = c.getLeftChild(c.getLeftChild(k));
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
      fLabel.add(getLabelID(c.getLabel(index)));

      index = c.getRightChild(c.getRightChild(k));
      fWord.add(getWordID(c.getWord(index)));
      fPos.add(getPosID(c.getPOS(index)));
      fLabel.add(getLabelID(c.getLabel(index)));
    }

    List<Integer> feature = new ArrayList<>(48);
    feature.addAll(fWord);
    feature.addAll(fPos);
    feature.addAll(fLabel);
    return feature;
  }

  private static final int POS_OFFSET = 18;
  private static final int DEP_OFFSET = 36;
  private static final int STACK_OFFSET = 6;
  private static final int STACK_NUMBER = 6;

  public int[] getFeatureArray(Configuration c) {
    int[] feature = new int[config.numTokens];  // positions 0-17 hold fWord, 18-35 hold fPos, 36-47 hold fLabel

    for (int j = 2; j >= 0; --j) {
      int index = c.getStack(j);
      feature[2-j] = getWordID(c.getWord(index));
      feature[POS_OFFSET + (2-j)] = getPosID(c.getPOS(index));
    }

    for (int j = 0; j <= 2; ++j) {
      int index = c.getBuffer(j);
      feature[3 + j] = getWordID(c.getWord(index));
      feature[POS_OFFSET + 3 + j] = getPosID(c.getPOS(index));
    }

    for (int j = 0; j <= 1; ++j) {
      int k = c.getStack(j);

      int index = c.getLeftChild(k);
      feature[STACK_OFFSET + j * STACK_NUMBER] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER] = getLabelID(c.getLabel(index));

      index = c.getRightChild(k);
      feature[STACK_OFFSET + j * STACK_NUMBER + 1] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 1] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 1] = getLabelID(c.getLabel(index));

      index = c.getLeftChild(k, 2);
      feature[STACK_OFFSET + j * STACK_NUMBER + 2] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 2] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 2] = getLabelID(c.getLabel(index));

      index = c.getRightChild(k, 2);
      feature[STACK_OFFSET + j * STACK_NUMBER + 3] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 3] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 3] = getLabelID(c.getLabel(index));

      index = c.getLeftChild(c.getLeftChild(k));
      feature[STACK_OFFSET + j * STACK_NUMBER + 4] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 4] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 4] = getLabelID(c.getLabel(index));

      index = c.getRightChild(c.getRightChild(k));
      feature[STACK_OFFSET + j * STACK_NUMBER + 5] = getWordID(c.getWord(index));
      feature[POS_OFFSET + STACK_OFFSET + j * STACK_NUMBER + 5] = getPosID(c.getPOS(index));
      feature[DEP_OFFSET + j * STACK_NUMBER + 5] = getLabelID(c.getLabel(index));
    }

    return feature;
  }
  
  public List<Integer> intArrays2List(int[] array){
	  List<Integer> retval = new ArrayList<>();
	  for(int a : array)
		  retval.add(a);
	  return retval;
  }
  
  /*
   * get local training examples
   * 
   */
  public Dataset genGreedyTrainExamples(List<CoreMap> sents, List<DependencyTree> trees) {
	    Dataset ret = new Dataset(config.numTokens, system.transitions.size());

	    Counter<Integer> tokPosCount = new IntCounter<>();
	    System.err.println(Config.SEPARATOR);
	    System.err.println("Generate training examples...");

	    for (int i = 0; i < sents.size(); ++i) {

	      if (i > 0) {
	        if (i % 1000 == 0)
	          System.err.print(i + " ");
	        if (i % 10000 == 0 || i == sents.size() - 1)
	          System.err.println();
	      }

	      if (trees.get(i).isProjective()) {
	        Configuration c = system.initialConfiguration(sents.get(i));

	        while (!system.isTerminal(c)) {
	          Pair<Integer, Integer> oraclePair = system.getHierarchicalOracle(c, trees.get(i));
	          
	          int[] actTypeLabel = system.getValidActType(c);
	          int[] depTypeLabel = system.getValidLabelGivenActType(c, oraclePair.first);
	          
	          actTypeLabel[oraclePair.first] = 1;
	          if(oraclePair.first != ParsingSystem.shiftActTypeID)
	        	  depTypeLabel[oraclePair.second] = 1;
	          
	          List<Integer> feature = getFeatures(c);

	          ret.addExample(feature, intArrays2List(actTypeLabel), 
	        		  depTypeLabel != null ? intArrays2List(depTypeLabel) : null);
	          
	          for (int j = 0; j < feature.size(); ++j)
	            tokPosCount.incrementCount(feature.get(j) * feature.size() + j);
	          system.apply(c, oraclePair.first, oraclePair.second);
	        }
	      }
	    }
	    System.err.println("#Train Examples: " + ret.n);

	    preComputed = new ArrayList<>(config.numPreComputed);
	    List<Integer> sortedTokens = Counters.toSortedList(tokPosCount, false);

	    preComputed = new ArrayList<>(sortedTokens.subList(0, Math.min(config.numPreComputed, sortedTokens.size())));

	    return ret;
	  }

  public Dataset genGlobalTrainExamples(List<CoreMap> sents, List<DependencyTree> trees) {
    Dataset ret = new Dataset(config.numTokens, system.transitions.size());

    Counter<Integer> tokPosCount = new IntCounter<>();
    System.err.println(Config.SEPARATOR);
    System.err.println("Generate training examples...");

    globalExamples = new ArrayList<GlobalExample>();
    
    
    for (int i = 0; i < sents.size(); ++i) {

    	List<Example> examples = new ArrayList<Example>();
    	List<Pair<Integer, Integer>> oracles = new ArrayList<Pair<Integer, Integer>>();
    	
      if (i > 0) {
        if (i % 1000 == 0)
          System.err.print(i + " ");
        if (i % 10000 == 0 || i == sents.size() - 1)
          System.err.println();
      }

      if (trees.get(i).isProjective()) {
        Configuration c = system.initialConfiguration(sents.get(i));

        while (!system.isTerminal(c)) {
        	
        	Pair<Integer, Integer> oraclePair = system.getHierarchicalOracle(c, trees.get(i));
        	
        	int[] actTypeLabel = system.getValidActType(c);
        	int[] depTypeLabel = system.getValidLabelGivenActType(c, oraclePair.first);
        	
        	actTypeLabel[oraclePair.first] = 1;
        	if(depTypeLabel != null) depTypeLabel[oraclePair.second] = 1;
        	
        	List<Integer> feature = getFeatures(c);
	          
          examples.add(new Example(feature, intArrays2List(actTypeLabel), 
        		  depTypeLabel == null ? null : intArrays2List(depTypeLabel)));
          oracles.add(oraclePair);
          for (int j = 0; j < feature.size(); ++j)
            tokPosCount.incrementCount(feature.get(j) * feature.size() + j);
          system.apply(c, oraclePair.first, oraclePair.second);
        }
      }
      
      globalExamples.add(new GlobalExample(sents.get(i), trees.get(i), examples, oracles));
      
    }
    System.err.println("#Train Examples: " + ret.n);

    preComputed = new ArrayList<>(config.numPreComputed);
    List<Integer> sortedTokens = Counters.toSortedList(tokPosCount, false);

    preComputed = new ArrayList<>(sortedTokens.subList(0, Math.min(config.numPreComputed, sortedTokens.size())));
    return ret;
  }

  /**
   * Generate unique integer IDs for all known words / part-of-speech
   * tags / dependency relation labels.
   *
   * All three of the aforementioned types are assigned IDs from a
   * continuous range of integers; all IDs 0 <= ID < n_w are word IDs,
   * all IDs n_w <= ID < n_w + n_pos are POS tag IDs, and so on.
   */
  private void generateIDs() {
    wordIDs = new HashMap<>();
    posIDs = new HashMap<>();
    labelIDs = new HashMap<>();

    int index = 0;
    for (String word : knownWords)
      wordIDs.put(word, (index++));
    for (String pos : knownPos)
      posIDs.put(pos, (index++));
    for (String label : knownLabels)
      labelIDs.put(label, (index++));
  }

  /**
   * Scan a corpus and store all words, part-of-speech tags, and
   * dependency relation labels observed. Prepare other structures
   * which support word / POS / label lookup at train- / run-time.
   */
  private void genDictionaries(List<CoreMap> sents, List<DependencyTree> trees) {
    // Collect all words (!), etc. in lists, tacking on one sentence
    // after the other
    List<String> word = new ArrayList<>();
    List<String> pos = new ArrayList<>();
    List<String> label = new ArrayList<>();

    for (CoreMap sentence : sents) {
      List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

      for (CoreLabel token : tokens) {
        word.add(token.word());
        pos.add(token.tag());
      }
    }

    String rootLabel = null;
    for (DependencyTree tree : trees)
      for (int k = 1; k <= tree.n; ++k)
        if (tree.getHead(k) == 0)
          rootLabel = tree.getLabel(k);
        else
          label.add(tree.getLabel(k));

    // Generate "dictionaries," possibly with frequency cutoff
    knownWords = Util.generateDict(word, config.wordCutOff);
    knownPos = Util.generateDict(pos);
    knownLabels = Util.generateDict(label);
    knownLabels.add(0, rootLabel);

    knownWords.add(0, Config.UNKNOWN);
    knownWords.add(1, Config.NULL);
    knownWords.add(2, Config.ROOT);

    knownPos.add(0, Config.UNKNOWN);
    knownPos.add(1, Config.NULL);
    knownPos.add(2, Config.ROOT);

    knownLabels.add(0, Config.NULL);
    generateIDs();

    System.out.println(Config.SEPARATOR);
    System.out.println("#Word: " + knownWords.size());
    System.out.println("#POS:" + knownPos.size());
    System.out.println("#Label: " + knownLabels.size());
  }

  public void writeModelFile(String modelFile) {
    try {
      double[][] W1 = classifier.getW1();
      double[] b1 = classifier.getb1();
      double[][] W2 = classifier.getW2();
      double[][] E = classifier.getE();
      double[][][] labelLayer = classifier.getLabelLayer();

      Writer output = IOUtils.getPrintWriter(modelFile);

      output.write("dict=" + knownWords.size() + "\n");
      output.write("pos=" + knownPos.size() + "\n");
      output.write("label=" + knownLabels.size() + "\n");
      output.write("embeddingSize=" + E[0].length + "\n");
      output.write("hiddenSize=" + b1.length + "\n");
      output.write("numTokens=" + (W1[0].length / E[0].length) + "\n");
      output.write("preComputed=" + preComputed.size() + "\n");

      int index = 0;

      // First write word / POS / label embeddings
      for (String word : knownWords) {
        output.write(word);
        for (int k = 0; k < E[index].length; ++k)
          output.write(" " + E[index][k]);
        output.write("\n");
        index = index + 1;
      }
      for (String pos : knownPos) {
        output.write(pos);
        for (int k = 0; k < E[index].length; ++k)
          output.write(" " + E[index][k]);
        output.write("\n");
        index = index + 1;
      }
      for (String label : knownLabels) {
        output.write(label);
        for (int k = 0; k < E[index].length; ++k)
          output.write(" " + E[index][k]);
        output.write("\n");
        index = index + 1;
      }

      // Now write classifier weights
      for (int j = 0; j < W1[0].length; ++j)
        for (int i = 0; i < W1.length; ++i) {
          output.write("" + W1[i][j]);
          if (i == W1.length - 1)
            output.write("\n");
          else
            output.write(" ");
        }
      for (int i = 0; i < b1.length; ++i) {
        output.write("" + b1[i]);
        if (i == b1.length - 1)
          output.write("\n");
        else
          output.write(" ");
      }
      for (int j = 0; j < W2[0].length; ++j)
        for (int i = 0; i < W2.length; ++i) {
          output.write("" + W2[i][j]);
          if (i == W2.length - 1)
            output.write("\n");
          else
            output.write(" ");
        }
      
      for(int i = 0; i < labelLayer.length; i++)
    	  for(int j = 0; j < labelLayer[0].length; j++)
    		  for(int k = 0; k < labelLayer[0][0].length; k++){
    			  output.write("" + labelLayer[i][j][k]);
    			  if(k == labelLayer[0][0].length - 1)
    				  output.write("\n");
    			  else
    				  output.write(" ");
    		  }

      // Finish with pre-computation info
      for (int i = 0; i < preComputed.size(); ++i) {
        output.write("" + preComputed.get(i));
        if ((i + 1) % 100 == 0 || i == preComputed.size() - 1)
          output.write("\n");
        else
          output.write(" ");
      }

      output.close();
    } catch (IOException e) {
      System.out.println(e);
    }
  }

  /**
   * Convenience method; see {@link #loadFromModelFile(String, java.util.Properties)}.
   *
   * @see #loadFromModelFile(String, java.util.Properties)
   */
  public static DependencyParser loadFromModelFile(String modelFile) {
    return loadFromModelFile(modelFile, null);
  }

  /**
   * Load a saved parser model.
   *
   * @param modelFile       Path to serialized model (may be GZipped)
   * @param extraProperties Extra test-time properties not already associated with model (may be null)
   *
   * @return Loaded and initialized (see {@link #initialize(boolean)} model
   */
  public static DependencyParser loadFromModelFile(String modelFile, Properties extraProperties) {
    DependencyParser parser = extraProperties == null ? new DependencyParser() : new DependencyParser(extraProperties);
    parser.loadModelFile(modelFile, false);
    return parser;
  }

  /** Load a parser model file, printing out some messages about the grammar in the file.
   *
   *  @param modelFile The file (classpath resource, etc.) to load the model from.
   */
  public void loadModelFile(String modelFile) {
    loadModelFile(modelFile, true);
  }

  private void loadModelFile(String modelFile, boolean verbose) {
    Timing t = new Timing();
    try {
      // System.err.println(Config.SEPARATOR);
      System.err.println("Loading depparse model file: " + modelFile + " ... ");
      String s;
      BufferedReader input = IOUtils.readerFromString(modelFile);

      int nDict, nPOS, nLabel;
      int eSize, hSize, nTokens, nPreComputed;
      nDict = nPOS = nLabel = eSize = hSize = nTokens = nPreComputed = 0;

      for (int k = 0; k < 7; ++k) {
        s = input.readLine();
        if (verbose) {
          System.err.println(s);
        }
        int number = Integer.parseInt(s.substring(s.indexOf('=') + 1));
        switch (k) {
          case 0:
            nDict = number;
            break;
          case 1:
            nPOS = number;
            break;
          case 2:
            nLabel = number;
            break;
          case 3:
            eSize = number;
            break;
          case 4:
            hSize = number;
            break;
          case 5:
            nTokens = number;
            break;
          case 6:
            nPreComputed = number;
            break;
          default:
            break;
        }
      }


      knownWords = new ArrayList<String>();
      knownPos = new ArrayList<String>();
      knownLabels = new ArrayList<String>();
      double[][] E = new double[nDict + nPOS + nLabel][eSize];
      String[] splits;
      int index = 0;

      for (int k = 0; k < nDict; ++k) {
        s = input.readLine();
        splits = s.split(" ");
        knownWords.add(splits[0]);
        for (int i = 0; i < eSize; ++i)
          E[index][i] = Double.parseDouble(splits[i + 1]);
        index = index + 1;
      }
      for (int k = 0; k < nPOS; ++k) {
        s = input.readLine();
        splits = s.split(" ");
        knownPos.add(splits[0]);
        for (int i = 0; i < eSize; ++i)
          E[index][i] = Double.parseDouble(splits[i + 1]);
        index = index + 1;
      }
      for (int k = 0; k < nLabel; ++k) {
        s = input.readLine();
        splits = s.split(" ");
        knownLabels.add(splits[0]);
        for (int i = 0; i < eSize; ++i)
          E[index][i] = Double.parseDouble(splits[i + 1]);
        index = index + 1;
      }
      generateIDs();

      double[][] W1 = new double[hSize][eSize * nTokens];
      for (int j = 0; j < W1[0].length; ++j) {
        s = input.readLine();
        splits = s.split(" ");
        for (int i = 0; i < W1.length; ++i)
          W1[i][j] = Double.parseDouble(splits[i]);
      }

      double[] b1 = new double[hSize];
      s = input.readLine();
      splits = s.split(" ");
      for (int i = 0; i < b1.length; ++i)
        b1[i] = Double.parseDouble(splits[i]);

      double[][] W2 = new double[ParsingSystem.nActTypeNum][hSize];
      for (int j = 0; j < W2[0].length; ++j) {
        s = input.readLine();
        splits = s.split(" ");
        for (int i = 0; i < W2.length; ++i)
          W2[i][j] = Double.parseDouble(splits[i]);
      }
      
      double[][][] labelLayer = new double[ParsingSystem.nActTypeNum][nLabel - 1][hSize]; // remove the NULL label
      for(int i = 0; i < labelLayer.length; i++)
    	  for(int j = 0; j < labelLayer[0].length; j++){
    		  s = input.readLine();
          	  splits = s.split(" ");
    		  for(int k = 0; k < labelLayer[0][0].length; k++){
    			  labelLayer[i][j][k] = Double.parseDouble(splits[k]);
    		  }
    	  }

      preComputed = new ArrayList<Integer>();
      while (preComputed.size() < nPreComputed) {
        s = input.readLine();
        splits = s.split(" ");
        for (String split : splits) {
          preComputed.add(Integer.parseInt(split));
        }
      }
      input.close();
      classifier = new Classifier(config, E, W1, b1, W2, labelLayer, preComputed);
      
      List<String> lDict = new ArrayList<String>(knownLabels);
      lDict.remove(0);
      ParsingSystem system = new ArcStandard(config.tlp, lDict, true);
      classifier.setParsingSystem(system);
      
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }

    // initialize the loaded parser
    initialize(verbose);
    t.done("Initializing dependency parser");
  }
  
//  /*
//   *   load model file in training
//   *   
//   *   remove new the classifier
//   */
//  private void loadModelFileInTraining(String modelFile, boolean verbose) {
//	    Timing t = new Timing();
//	    try {
//	      // System.err.println(Config.SEPARATOR);
//	      System.err.println("Loading depparse model file: " + modelFile + " ... ");
//	      String s;
//	      BufferedReader input = IOUtils.readerFromString(modelFile);
//
//	      int nDict, nPOS, nLabel;
//	      int eSize, hSize, nTokens, nPreComputed;
//	      nDict = nPOS = nLabel = eSize = hSize = nTokens = nPreComputed = 0;
//
//	      for (int k = 0; k < 7; ++k) {
//	        s = input.readLine();
//	        if (verbose) {
//	          System.err.println(s);
//	        }
//	        int number = Integer.parseInt(s.substring(s.indexOf('=') + 1));
//	        switch (k) {
//	          case 0:
//	            nDict = number;
//	            break;
//	          case 1:
//	            nPOS = number;
//	            break;
//	          case 2:
//	            nLabel = number;
//	            break;
//	          case 3:
//	            eSize = number;
//	            break;
//	          case 4:
//	            hSize = number;
//	            break;
//	          case 5:
//	            nTokens = number;
//	            break;
//	          case 6:
//	            nPreComputed = number;
//	            break;
//	          default:
//	            break;
//	        }
//	      }
//
//
//	      knownWords = new ArrayList<String>();
//	      knownPos = new ArrayList<String>();
//	      knownLabels = new ArrayList<String>();
//	      double[][] E = classifier.getE();
//	      String[] splits;
//	      int index = 0;
//
//	      for (int k = 0; k < nDict; ++k) {
//	        s = input.readLine();
//	        splits = s.split(" ");
//	        knownWords.add(splits[0]);
//	        for (int i = 0; i < eSize; ++i)
//	          E[index][i] = Double.parseDouble(splits[i + 1]);
//	        index = index + 1;
//	      }
//	      for (int k = 0; k < nPOS; ++k) {
//	        s = input.readLine();
//	        splits = s.split(" ");
//	        knownPos.add(splits[0]);
//	        for (int i = 0; i < eSize; ++i)
//	          E[index][i] = Double.parseDouble(splits[i + 1]);
//	        index = index + 1;
//	      }
//	      for (int k = 0; k < nLabel; ++k) {
//	        s = input.readLine();
//	        splits = s.split(" ");
//	        knownLabels.add(splits[0]);
//	        for (int i = 0; i < eSize; ++i)
//	          E[index][i] = Double.parseDouble(splits[i + 1]);
//	        index = index + 1;
//	      }
//	      generateIDs();
//
//	      double[][] W1 = classifier.getW1();
//	      for (int j = 0; j < W1[0].length; ++j) {
//	        s = input.readLine();
//	        splits = s.split(" ");
//	        for (int i = 0; i < W1.length; ++i)
//	          W1[i][j] = Double.parseDouble(splits[i]);
//	      }
//
//	      double[] b1 = classifier.getb1();
//	      s = input.readLine();
//	      splits = s.split(" ");
//	      for (int i = 0; i < b1.length; ++i)
//	        b1[i] = Double.parseDouble(splits[i]);
//
//	      double[][] W2 = classifier.getW2();
//	      for (int j = 0; j < W2[0].length; ++j) {
//	        s = input.readLine();
//	        splits = s.split(" ");
//	        for (int i = 0; i < W2.length; ++i)
//	          W2[i][j] = Double.parseDouble(splits[i]);
//	      }
//
//	      preComputed = new ArrayList<Integer>();
//	      while (preComputed.size() < nPreComputed) {
//	        s = input.readLine();
//	        splits = s.split(" ");
//	        for (String split : splits) {
//	          preComputed.add(Integer.parseInt(split));
//	        }
//	      }
//	      input.close();
//	      
//	      Map<Integer, Integer> preMap = new HashMap<>();
//	      for (int i = 0; i < preComputed.size(); ++i)
//	        preMap.put(preComputed.get(i), i);
//	      
//	      classifier.setPreMap(preMap);
//	    } catch (IOException e) {
//	      throw new RuntimeIOException(e);
//	    }
//
//	    // initialize the loaded parser
//	    initialize(verbose);
//	    t.done("Initializing dependency parser");
//	  }

  // TODO this should be a function which returns the embeddings array + embedID
  // otherwise the class needlessly carries around the extra baggage of `embeddings`
  // (never again used) for the entire training process
  private void readEmbedFile(String embedFile) {
    embedID = new HashMap<String, Integer>();
    if (embedFile == null)
      return;
    BufferedReader input = null;
    try {
      input = IOUtils.readerFromString(embedFile);
      List<String> lines = new ArrayList<String>();
      for (String s; (s = input.readLine()) != null; ) {
        lines.add(s);
      }

      int nWords = lines.size();
      String[] splits = lines.get(0).split("\\s+");

      int dim = splits.length - 1;
      embeddings = new double[nWords][dim];
      System.err.println("Embedding File " + embedFile + ": #Words = " + nWords + ", dim = " + dim);

      //TODO: how if the embedding dim. does not match..?
      if (dim != config.embeddingSize)
        System.err.println("ERROR: embedding dimension mismatch");

      for (int i = 0; i < lines.size(); ++i) {
        splits = lines.get(i).split("\\s+");
        embedID.put(splits[0], i);
        for (int j = 0; j < dim; ++j)
          embeddings[i][j] = Double.parseDouble(splits[j + 1]);
      }
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(input);
    }
  }

  /**
   * Train a new dependency parser model.
   *
   * @param trainFile Training data
   * @param devFile Development data (used for regular UAS evaluation
   *                of model)
   * @param modelFile String to which model should be saved
   * @param embedFile File containing word embeddings for words used in
   *                  training corpus
   */
  public void train(String trainFile, String devFile, String modelFile, String embedFile) {
    System.err.println("Train File: " + trainFile);
    System.err.println("Dev File: " + devFile);
    System.err.println("Model File: " + modelFile);
    System.err.println("Embedding File: " + embedFile);

    List<CoreMap> trainSents = new ArrayList<>();
    List<DependencyTree> trainTrees = new ArrayList<DependencyTree>();
    Util.loadConllFile(trainFile, trainSents, trainTrees);
    Util.printTreeStats("Train", trainTrees);

    List<CoreMap> devSents = new ArrayList<CoreMap>();
    List<DependencyTree> devTrees = new ArrayList<DependencyTree>();
    if (devFile != null) {
      Util.loadConllFile(devFile, devSents, devTrees);
      Util.printTreeStats("Dev", devTrees);
    }
    genDictionaries(trainSents, trainTrees);

    //NOTE: remove -NULL-, and the pass it to ParsingSystem
    List<String> lDict = new ArrayList<String>(knownLabels);
    lDict.remove(0);
    system = new ArcStandard(config.tlp, lDict, true);

    // Initialize a classifier; prepare for training
    setupClassifierForTraining(trainSents, trainTrees, embedFile);
    classifier.setParser(this);
    
    /**
     *   construct the global training examples
     */
    if(trainSents.size() != trainTrees.size()){
    	System.err.println("The sizes of sents and trees are not consistent!");
    
    	if (devFile != null) {
            // Redo precomputation with updated weights. This is only
            // necessary because we're updating weights -- for normal
            // prediction, we just do this once in #initialize
            classifier.preCompute();

            List<List<DependencyTree>> predictedBeam;
            List<DependencyTree> predicated = new ArrayList<>();
            if(config.globalTraining){
            	predictedBeam = devSents.stream().map(this::predictInnerWithBeam).collect(toList());
            	for(int i = 0; i<predictedBeam.size(); i++)
            		predicated.add(predictedBeam.get(i).get(0));
            }
            else
            	predicated = devSents.stream().map(this::predictInner).collect(toList());
            
			//      	  system.evaluateOracle(devSents, predictedBeam, devTrees);
      	  

            double uas = system.getUASScore(devSents, predicated, devTrees);
            System.err.println("base model UAS: " + uas);

          }
    }
//    List<GlobalExample> globalExamples = new ArrayList<GlobalExample>();
//    for(int i = 0; i<trainSents.size(); i++)
//    	globalExamples.add(new GlobalExample(trainSents.get(i), trainTrees.get(i) ));
    

    System.err.println(Config.SEPARATOR);
    config.printParameters();

    long startTime = System.currentTimeMillis();
    /**
     * Track the best UAS performance we've seen.
     */
    double bestUAS = 0;

    for (int iter = 0; iter < config.maxIter; ++iter) {
      System.err.println("##### Iteration " + iter);

      if(config.globalTraining){ //global traing with beam search
    	  if( config.sgdTraining ){
    		  
    	  }
    	  else{
    		  
    		  Classifier.Cost cost = classifier.computeGlobalCostFunction(globalExamples, config.batchSize, config.regParameter, config.dropProb,
    				  config.nBeam, config.dMargin);
    		  System.err.println("Cost = " + cost.getCost() + ", Correct(%) = " + cost.getPercentCorrect());
    		  classifier.takeAdaGradientStep(cost, config.adaAlpha, config.adaEps);
    	  }
      }
      else{
    	  Classifier.Cost cost = classifier.computeGreedyCostFunction(config.batchSize, 
    			  config.regParameter, config.dropProb);
		  System.err.println("Cost = " + cost.getCost() + ", Correct(%) = " + cost.getPercentCorrect());
		  classifier.takeAdaGradientStep(cost, config.adaAlpha, config.adaEps);
      }
      System.err.println("Elapsed Time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");

      // UAS evaluation
      if (devFile != null && iter % config.evalPerIter == 0) {
        // Redo precomputation with updated weights. This is only
        // necessary because we're updating weights -- for normal
        // prediction, we just do this once in #initialize
    	  
    	  classifier.preCompute();
    	  
    	  List<List<DependencyTree>> predictedBeam;
          List<DependencyTree> predicated = new ArrayList<>();
          if(config.globalTraining){
          	predictedBeam = devSents.stream().map(this::predictInnerWithBeam).collect(toList());
          	for(int i = 0; i<predictedBeam.size(); i++)
          		predicated.add(predictedBeam.get(i).get(0));
          }
          else
          	predicated = devSents.stream().map(this::predictInner).collect(toList());
          
          double uas = system.getUASScore(devSents, predicated, devTrees);
          System.err.println("base model UAS: " + uas);
          
        if (config.saveIntermediate && uas > bestUAS) {
          System.err.printf("Exceeds best previous UAS of %f. Saving model file..%n", bestUAS);

          bestUAS = uas;
          writeModelFile(modelFile);
        }
      }

      // Clear gradients
      if (config.clearGradientsPerIter > 0 && iter % config.clearGradientsPerIter == 0) {
        System.err.println("Clearing gradient histories..");
        classifier.clearGradientHistories();
      }
    }

    classifier.finalizeTraining();

  }

  /**
   * @see #train(String, String, String, String)
   */
  public void train(String trainFile, String devFile, String modelFile) {
    train(trainFile, devFile, modelFile, null);
  }

  /**
   * @see #train(String, String, String, String)
   */
  public void train(String trainFile, String modelFile) {
    train(trainFile, null, modelFile);
  }

  /**
   * Prepare a classifier for training with the given dataset.
   */
  private void setupClassifierForTraining(List<CoreMap> trainSents, List<DependencyTree> trainTrees, String embedFile) {
    double[][] E = new double[knownWords.size() + knownPos.size() + knownLabels.size()][config.embeddingSize];
    double[][] W1 = new double[config.hiddenSize][config.embeddingSize * config.numTokens];
    double[] b1 = new double[config.hiddenSize];
    double[][] W2 = new double[system.nActTypeNum][config.hiddenSize];
    double[][][] labelLayer = new double[system.nActTypeNum][system.labels.size()][config.hiddenSize];

    // Randomly initialize weight matrices / vectors
    Random random = Util.getRandom();
    for (int i = 0; i < W1.length; ++i)
      for (int j = 0; j < W1[i].length; ++j)
        W1[i][j] = random.nextDouble() * 2 * config.initRange - config.initRange;

    for (int i = 0; i < b1.length; ++i)
      b1[i] = random.nextDouble() * 2 * config.initRange - config.initRange;

    for (int i = 0; i < W2.length; ++i)
      for (int j = 0; j < W2[i].length; ++j)
        W2[i][j] = random.nextDouble() * 2 * config.initRange - config.initRange;

    for (int i = 0; i < labelLayer.length; ++i)
        for (int j = 0; j < labelLayer[i].length; ++j)
        	for(int k = 0; k < labelLayer[i][j].length; k++)
        		labelLayer[i][j][k] = random.nextDouble() * 2 * config.initRange - config.initRange;
    // Read embeddings into `embedID`, `embeddings`
    readEmbedFile(embedFile);

    // Try to match loaded embeddings with words in dictionary
    int foundEmbed = 0;
    for (int i = 0; i < E.length; ++i) {
      int index = -1;
      if (i < knownWords.size()) {
        String str = knownWords.get(i);
        //NOTE: exact match first, and then try lower case..
        if (embedID.containsKey(str)) index = embedID.get(str);
        else if (embedID.containsKey(str.toLowerCase())) index = embedID.get(str.toLowerCase());
      }

      if (index >= 0) {
        ++foundEmbed;
        for (int j = 0; j < E[i].length; ++j)
          E[i][j] = embeddings[index][j];
      } else {
        for (int j = 0; j < E[i].length; ++j)
          E[i][j] = random.nextDouble() * config.initRange * 2 - config.initRange;
      }
    }
    System.err.println("Found embeddings: " + foundEmbed + " / " + knownWords.size());

    Dataset trainSet;
    if(config.globalTraining)
    	trainSet = genGlobalTrainExamples(trainSents, trainTrees);
    else
    	trainSet = genGreedyTrainExamples(trainSents, trainTrees);
    
    classifier = new Classifier(config, trainSet, E, W1, b1, W2, labelLayer, preComputed);
    classifier.setParsingSystem(system);
  }

  /**
   * Determine the dependency parse of the given sentence.
   * <p>
   * This "inner" method returns a structure unique to this package; use {@link #predict(edu.stanford.nlp.util.CoreMap)}
   * for general parsing purposes.
   */
  private DependencyTree predictInner(CoreMap sentence) {
	  
    Configuration c = system.initialConfiguration(sentence);
    while (!system.isTerminal(c)) {
    	
      Pair<Integer, Integer> optActPair = classifier.computeHierarchicalScore(getFeatureArray(c), c);

      system.apply(c, optActPair.first, optActPair.second);
    }
    return c.tree;
  }

  /**
   * Determine the dependency parse of the given sentence using the loaded model.
   * You must first load a parser before calling this method.
   *
   * @throws java.lang.IllegalStateException If parser has not yet been loaded and initialized
   *         (see {@link #initialize(boolean)}
   */
  public GrammaticalStructure predict(CoreMap sentence) {
    if (system == null)
      throw new IllegalStateException("Parser has not been  " +
          "loaded and initialized; first load a model.");

    DependencyTree result = predictInner(sentence);

    // The rest of this method is just busy-work to convert the
    // package-local representation into a CoreNLP-standard
    // GrammaticalStructure.

    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    List<TypedDependency> dependencies = new ArrayList<>();

    IndexedWord root = new IndexedWord(new Word("ROOT"));
    root.set(CoreAnnotations.IndexAnnotation.class, 0);

    for (int i = 1; i <= result.n; i++) {
      int head = result.getHead(i);
      String label = result.getLabel(i);

      IndexedWord thisWord = new IndexedWord(tokens.get(i - 1));
      IndexedWord headWord = head == 0 ? root
                                       : new IndexedWord(tokens.get(head - 1));

      GrammaticalRelation relation = head == 0
                                     ? GrammaticalRelation.ROOT
                                     : new GrammaticalRelation(language, label, null,
                                         GrammaticalRelation.DEPENDENT);

      dependencies.add(new TypedDependency(relation, headWord, thisWord));
    }

    // Build GrammaticalStructure
    // TODO ideally submodule should just return GrammaticalStructure
    TreeGraphNode rootNode = new TreeGraphNode(root);
    return new EnglishGrammaticalStructure(dependencies, rootNode);
  }

  /**
   * Convenience method for {@link #predict(edu.stanford.nlp.util.CoreMap)}. The tokens of the provided sentence must
   * also have tag annotations (the parser requires part-of-speech tags).
   *
   * @see #predict(edu.stanford.nlp.util.CoreMap)
   */
  public GrammaticalStructure predict(List<? extends HasWord> sentence) {
    CoreLabel sentenceLabel = new CoreLabel();
    List<CoreLabel> tokens = new ArrayList<>();

    int i = 1;
    for (HasWord wd : sentence) {
      CoreLabel label;
      if (wd instanceof CoreLabel) {
        label = (CoreLabel) wd;
        if (label.tag() == null)
          throw new IllegalArgumentException("Parser requires words " +
              "with part-of-speech tag annotations");
      } else {
        label = new CoreLabel();
        label.setValue(wd.word());
        label.setWord(wd.word());

        if (!(wd instanceof HasTag))
          throw new IllegalArgumentException("Parser requires words " +
              "with part-of-speech tag annotations");

        label.setTag(((HasTag) wd).tag());
      }

      label.setIndex(i);
      i++;

      tokens.add(label);
    }

    sentenceLabel.set(CoreAnnotations.TokensAnnotation.class, tokens);

    return predict(sentenceLabel);
  }

  //TODO: support sentence-only files as input

  /** Run the parser in the modelFile on a testFile and perhaps save output.
   *
   *  @param testFile File to parse. In CoNLL-X format. Assumed to have gold answers included.
   *  @param outFile File to write results to in CoNLL-X format.  If null, no output is written
   *  @return The LAS score on the dataset
   */
  public double testCoNLL(String testFile, String outFile) {
    System.err.println("Test File: " + testFile);
    Timing timer = new Timing();
    List<CoreMap> testSents = new ArrayList<>();
    List<DependencyTree> testTrees = new ArrayList<DependencyTree>();
    Util.loadConllFile(testFile, testSents, testTrees);
    // count how much to parse
    int numWords = 0;
    int numSentences = 0;
    for (CoreMap testSent : testSents) {
      numSentences += 1;
      numWords += testSent.get(CoreAnnotations.TokensAnnotation.class).size();
    }

    List<DependencyTree> predicted = null;
    
    if(config.globalTraining){
    	
    }
    else
    	predicted = testSents.stream().map(this::predictInner).collect(toList());
    	
    Map<String, Double> result = system.evaluate(testSents, predicted, testTrees);
    double lasNoPunc = result.get("LASwoPunc");
    System.err.printf("UAS = %.4f%n", result.get("UASwoPunc"));
    System.err.printf("LAS = %.4f%n", lasNoPunc);
    long millis = timer.stop();
    double wordspersec = numWords / (((double) millis) / 1000);
    double sentspersec = numSentences / (((double) millis) / 1000);
    System.err.printf("%s tagged %d words in %d sentences in %.1fs at %.1f w/s, %.1f sent/s.%n",
            StringUtils.getShortClassName(this), numWords, numSentences, millis / 1000.0, wordspersec, sentspersec);

    if (outFile != null) {
        Util.writeConllFile(outFile, testSents, predicted);
    }
    return lasNoPunc;
  }

  private void parseTextFile(BufferedReader input, PrintWriter output) {
    DocumentPreprocessor preprocessor = new DocumentPreprocessor(input);
    preprocessor.setSentenceFinalPuncWords(config.tlp.sentenceFinalPunctuationWords());
    preprocessor.setEscaper(config.escaper);
    preprocessor.setSentenceDelimiter(config.sentenceDelimiter);
    preprocessor.setTokenizerFactory(config.tlp.getTokenizerFactory());

    Timing timer = new Timing();

    MaxentTagger tagger = new MaxentTagger(config.tagger);
    List<List<TaggedWord>> tagged = new ArrayList<>();
    for (List<HasWord> sentence : preprocessor) {
      tagged.add(tagger.tagSentence(sentence));
    }

    System.err.printf("Tagging completed in %.2f sec.%n",
        timer.stop() / 1000.0);

    timer.start();

    int numSentences = 0;
    for (List<TaggedWord> taggedSentence : tagged) {
      GrammaticalStructure parse = predict(taggedSentence);

      Collection<TypedDependency> deps = parse.typedDependencies();
      for (TypedDependency dep : deps)
        output.println(dep);
      output.println();

      numSentences++;
    }

    long millis = timer.stop();
    double seconds = millis / 1000.0;
    System.err.printf("Parsed %d sentences in %.2f seconds (%.2f sents/sec).%n",
        numSentences, seconds, numSentences / seconds);
  }

  /**
   * Prepare for parsing after a model has been loaded.
   */
  private void initialize(boolean verbose) {
    if (knownLabels == null)
      throw new IllegalStateException("Model has not been loaded or trained");

    // NOTE: remove -NULL-, and then pass the label set to the ParsingSystem
    List<String> lDict = new ArrayList<>(knownLabels);
    lDict.remove(0);

    system = new ArcStandard(config.tlp, lDict, verbose);

    // Pre-compute matrix multiplications
    if (config.numPreComputed > 0) {
      classifier.preCompute();
    }
  }
  
  /**
   * compute the oracle by gold action, and opt label given gold action
   * 
   * @param sentence
   * @param goldTree
   * @return
   */
  private DependencyTree computeOracle(CoreMap sentence, DependencyTree goldTree) {
	  int currentDepth = 0;
	  int oracleDepth = config.nOracleDepth;
	  
	  Configuration c = system.initialConfiguration(sentence);
	  while (!system.isTerminal(c)) {
		  
		  int[] feature = getFeatureArray(c);
		  Pair<Integer, Integer> optActPair = classifier.computeHierarchicalScore(feature, c);
		  
		  int optActType = optActPair.first;
		  int optDepType = optActPair.second;
		  
		  Pair<Integer, Integer> goldTrans = null;
		  if(currentDepth < oracleDepth)
			  goldTrans = system.getHierarchicalOracle(c, goldTree);
		  
		  if(currentDepth < oracleDepth && optActType != goldTrans.first){
			  currentDepth++;
			  optActType = goldTrans.first;
			  if( optActType != system.shiftActTypeID )
				  optDepType = classifier.getOptDepTypeGivenActType(feature, c, optActType);
		  }
		  
		  system.apply(c, optActType, optDepType);
	  }
	  return c.tree;
  }
  
  public double getOracle(String testFile, String outFile) {
	    System.err.println("Test Oracle File: " + testFile);
	    Timing timer = new Timing();
	    List<CoreMap> testSents = new ArrayList<>();
	    List<DependencyTree> testTrees = new ArrayList<DependencyTree>();
	    Util.loadConllFile(testFile, testSents, testTrees);
	    // count how much to parse
	    int numWords = 0;
	    int numSentences = 0;
	    for (CoreMap testSent : testSents) {
	      numSentences += 1;
	      numWords += testSent.get(CoreAnnotations.TokensAnnotation.class).size();
	    }

	    List<DependencyTree> predicted = new ArrayList<DependencyTree>();
	    for(int i = 0; i<testSents.size(); i++){
	    	predicted.add(computeOracle(testSents.get(i), testTrees.get(i)));
	    }

	    Map<String, Double> result = system.evaluate(testSents, predicted, testTrees);
	    double lasNoPunc = result.get("LASwoPunc");
	    System.err.printf("UAS = %.4f%n", result.get("UASwoPunc"));
	    System.err.printf("LAS = %.4f%n", lasNoPunc);
	    long millis = timer.stop();
	    double wordspersec = numWords / (((double) millis) / 1000);
	    double sentspersec = numSentences / (((double) millis) / 1000);
	    System.err.printf("%s tagged %d words in %d sentences in %.1fs at %.1f w/s, %.1f sent/s.%n",
	            StringUtils.getShortClassName(this), numWords, numSentences, millis / 1000.0, wordspersec, sentspersec);

	    if (outFile != null) {
	        Util.writeConllFile(outFile, testSents, predicted);
	    }
	    return lasNoPunc;
	  }
  
  public void outputWrongActProb(String testFile, String outFile) throws IOException{
	  System.err.println("Test Oracle File: " + testFile);
	  Timing timer = new Timing();
	  List<CoreMap> testSents = new ArrayList<>();
	  List<DependencyTree> testTrees = new ArrayList<DependencyTree>();
	  Util.loadConllFile(testFile, testSents, testTrees);
	  PrintWriter output = IOUtils.getPrintWriter(outFile);
	  
	  for(int i = 0; i<testSents.size(); i++){
		  wrongActAnalysis(testSents.get(i), testTrees.get(i), output);
	  }
	  
	  output.close();
	  
	    
  }


/**
 * output the probability distribution of a wrong exec action,
 * and then given it gold action for next wrong action analysis
 * 
 * @param coreMap
 * @param dependencyTree
 * @param output
 */
private void wrongActAnalysis(CoreMap sentence, DependencyTree goldTree, PrintWriter output) {
	  
	
	int wrong = -1;
	int right = 1;
	  Configuration c = system.initialConfiguration(sentence);
	  while (!system.isTerminal(c)) {
		  
		  int[] feature = getFeatureArray(c);
//		  Pair<Integer, Integer> optActPair = classifier.computeHierarchicalScore(feature, c);
		  
		  Pair<Integer, Integer> goldTrans = null;
		  goldTrans = system.getHierarchicalOracle(c, goldTree);
		  
		  Pair<Integer, Double> result = classifier.getOptActType(feature, c);
		  int optActType = result.first;
		  int optDepType = classifier.getOptDepTypeGivenActType(feature, c, optActType);

		  if(optActType != goldTrans.first){
			  
			  /*
			   * output the analysis information
			   * format: wrong or right + wrong action ID + gold action ID + action probability
			   *  
			   */
			  output.println(wrong + "\t" + optActType + "\t" + goldTrans.first + "\t" + result.second);
			  optActType = goldTrans.first;
			  optDepType = classifier.getOptDepTypeGivenActType(feature, c, optActType);
		  }
		  else{
			  output.println(right + "\t" + optActType + "\t" + goldTrans.first + "\t" + result.second);
		  }
		  
		  system.apply(c, optActType, optDepType);
	  }
	  
	  output.flush();
}

/**
 * given a parsing state and an action type, 
 * return the opt final action state
 *  
 * @param state
 * @param givenActType
 * @param sentence
 * @param keepGold
 * @return
 */
public HierarchicalDepState partialGreedyParser(
		HierarchicalDepState initialState, 
		int givenActType, 
		CoreMap sentence, 
		boolean keepGold){
	
	/*
	 * Initial the parsing state
	 */
	HierarchicalDepState pState = null;
	if(initialState != null){
		Configuration c = system.initialConfiguration(sentence);
		pState = new HierarchicalDepState(c, -1, -1, 0.0, 0.0, null, true);
	}
	else
		pState = initialState;
	
	while (!system.isTerminal(pState.c)) {
		
		/*
		 * apply the given action for a given state
		 */
		if( initialState != null && keepGold ){  //exec the given action, the scores have been computed already 
			
			/*
			 * compute the optimal deptype and expand the state
			 */
			Pair<Integer, Double> optDepTypeAndProb = 
					classifier.getOptDepTypeAndProb(pState, givenActType);
			double actTypeLogProb = pState.actTypeScore + 
					Math.log(pState.actTypeDistribution[givenActType]);
			double depTypeLogProb = pState.depTypeScore + Math.log(optDepTypeAndProb.second);
			
			HierarchicalDepState newState = new HierarchicalDepState(
					pState.c, 
					givenActType, 
					optDepTypeAndProb.first, 
					actTypeLogProb, 
					depTypeLogProb, 
					pState, 
					false);
			pState = newState;
			//generate the next state
			keepGold = false;  //only keep one action
			continue;  //exec the given action, continue directly
		}
		
		
		/*
		 * apply the optimal actions until terminates
		 */
		
		/*
		 * compute the relevance paras for a given state
		 */
		int[] actTypeLabel = system.getValidActType(pState.c);
		int[] feature = getFeatureArray(pState.c);
		HiddenLayer hidden = classifier.getHidden(false, pState.featureArray, pState.c);
		pState.setParas(hidden, actTypeLabel, feature);
		
		/*
		 * compute the hierarchical output layer
		 */
		Pair<Integer, double[]> optActTypeAndLayer = classifier.getOptActTypeAndLayer(pState);
		pState.setActTypeDistribution(optActTypeAndLayer.second);
		Pair<Integer, Double> optDepTypeAndProb = classifier.getOptDepTypeAndProb(pState, optActTypeAndLayer.first);
		
		/*
		 * construct the new state
		 * 
		 * #NOTE the score in the state is log() score, but the score array is
		 *       still probability, for convenient compare
		 */
		double actTypeLogProb = pState.actTypeScore + Math.log(optActTypeAndLayer.second[optActTypeAndLayer.first]);
		double depTypeLogProb = pState.depTypeScore + Math.log(optDepTypeAndProb.second);
		HierarchicalDepState newState = new HierarchicalDepState(
				pState.c, 
				optActTypeAndLayer.first, 
				optDepTypeAndProb.first, 
				actTypeLogProb, 
				depTypeLogProb, 
				pState, 
				false);
		
		pState = newState;
		pState.StateApply(system);
		
	}
	
	return pState;
}

/**
 *   Best-First Revise Module
 *   
 *   Revise the greedy result of parser with a best-first framework
 *   The best-first queue is constructed with action margin and tree model score
 * 
 */
private List<Pair<Double, DependencyTree>> bestfirstRevise(CoreMap sent) {

	List<Pair<Double, DependencyTree>> retval = new ArrayList<Pair<Double,DependencyTree>>();

	/*
	 *   Given a parsing state, get top config.nMaxReviseActNum revisedItem from 
	 *   revisedItemsFromOneState queue and insert them into the queue according to the 
	 *   product of margin and average of log probability of a complete parsing state
	 *   
	 *   In each step, we peek one revised item form the queue and get the revised CFGTree.
	 *   Then insert the revised CFGTree into the revisedTrees queue.
	 *   
	 *   In the end, when the revisedTrees queue euqal the config.nMaxN, stop and return 
	 *   these revised CFGTrees.
	 */

	// priority queue from small to large
	PriorityQueue<RevisedState> queue = new PriorityQueue<RevisedState>();
	// priority queue of revised parsing tree
	PriorityQueue<ScoredDepTree> revisedTrees = new PriorityQueue<ScoredDepTree>();
	// priority queue of revised items from one complete parsing state
	PriorityQueue<RevisedState> revisedItemsFromOneState = new PriorityQueue<RevisedState>();
	
	int transionNum = system.transitions.size();
	
	// get the initial result
	HierarchicalDepState initState = partialGreedyParser(null, -1, sent, false); 
	
	//add the greedy parser result first.
	retval.add(new Pair<Double, DependencyTree>(initState.score, initState.c.tree ));
	boolean firstRevise = true;
	
	//loop until the revised tree size to nMaxN
	while(revisedTrees.size() < config.nMaxN){
		
		/*
		 * first revise from the optimal tree by the greedy baseline parser
		 */
		if(firstRevise){  //revise from the greedy classifier output
			
			HierarchicalDepState state = initState; 
			double initStateScore = state.score;
			
			state =state.lastState; //from last state of the final state
								    //because the final state do not need devise
			revisedItemsFromOneState.clear();
			
			//get the acts in the greedy state
			while(state != null){
				
				int[] label = state.actTypeLabel;
				double[] scores =state.actTypeDistribution;
				int bestAct = state.actType;
				
				for(int j = 0; j < ParsingSystem.nActTypeNum; j++){
					double margin = scores[bestAct] - scores[j];
					
					//if action margin is larger than max margin, or is the best or valid
					//action, just skip
					if(margin < config.dMargin &&
							label[j] == 0)
						revisedItemsFromOneState.add( new RevisedState(new ReviseItem(-1, j, margin), state, initStateScore) );
					
				}
				
				state = state.lastState;
			}
			
			firstRevise = false;
		}
		else{  //revise from already revised parsing state
			
			if(queue.size() == 0) //no candidate in the queue
				break;
			
			RevisedState rs = queue.poll();
			//generate the new state and insert the revised tree to revisedTree Queue
				
			HierarchicalDepState state = partialGreedyParser(rs.state, rs.item.reviseActID, sent, true);
			double initStateScore = state.score;
			revisedTrees.add(new ScoredDepTree(state.c.tree, initStateScore));
		
			revisedItemsFromOneState.clear();
			state = state.lastState; //from last state to second last state!
									 //because the last state do not need devise
			
			//get the inherit revision candidate
			while(state != rs.state){ //till the previous revised state
				
				int[] label = state.actTypeLabel;
				double[] scores =state.actTypeDistribution;
				int aptAct = state.actType;
				
				for(int j = 0; j < ParsingSystem.nActTypeNum; j++){
					double margin = scores[aptAct] - scores[j];
					
					//if action margin is larger than max margin, or is the best or valid
					//action, just skip
					if (j == aptAct) 
						continue;
					
					if(margin < config.dMargin)
						revisedItemsFromOneState.add( new RevisedState(new ReviseItem(-1, j, margin), state, initStateScore) );
				}
				
				state = state.lastState;
			}
			
		
		}
		
		//add the best n reviseItem to revisedState queue
		for(int k = 0; k < config.nMaxReviseActNum; k++){
			if(revisedItemsFromOneState.size()==0)
				break;
			
			queue.add(revisedItemsFromOneState.poll());
		}
	}
	
	/*
	 * get the k-best result from the queue and return
	 */
	for(int i = 0; i < config.nMaxN; i++){
		
		if(revisedTrees.size() == 0)
			break;
		
		ScoredDepTree st = revisedTrees.poll();
		retval.add(new Pair<Double, DependencyTree>(st.score, st.tree));
	}
	return retval;
}

/**
 * sample revised oracle
 * 
 * @param testFile
 * @param outputFile
 * @param outputGoldFile
 * @throws IOException
 */
private void getNBest(String testFile, String outputFile) throws IOException {

	PrintWriter output = IOUtils.getPrintWriter(outputFile);
	List<List<Pair<Double, DependencyTree>>> outputNBest = new ArrayList<List<Pair<Double,DependencyTree>>>();
	List<CoreMap> testSents = new ArrayList<>();
	
	List<DependencyTree> bestTrees = new ArrayList<DependencyTree>();
	List<DependencyTree> worstTrees = new ArrayList<DependencyTree>();
	List<DependencyTree> goldTrees = new ArrayList<DependencyTree>();
	List<DependencyTree> originalTrees = new ArrayList<DependencyTree>();
	
		
	System.err.println("Begin to revise file");
	testSents.clear();
	bestTrees.clear();
	worstTrees.clear();
	
	Util.loadConllFile(testFile, testSents, goldTrees);
	
	for(int j = 0; j<testSents.size(); j++){
		
		System.err.println("Begin to revise sentence "+j);
		
		CoreMap sent = testSents.get(j);
		
		if(config.bBeamNBest){
//			outputNBest.add(beamDecode(gTree, sent));
		}
		else if(config.bBestFirstRevise)
			outputNBest.add(bestfirstRevise(sent));
		
	}
	
	// oracle computing
	double bestF1 = 0;
	double worstF1 = 0;
	DependencyTree bestTree = null;
	DependencyTree worstTree = null;
	
	//output
	for(int j = 0; j<testSents.size(); j++){
		
		List<Pair<Double, DependencyTree>> nbest = outputNBest.get(j);
		output.flush();
		originalTrees.add(nbest.get(0).second);  //the original tree is always
		//the first one of returns
		
		for(int k = 0; k<nbest.size(); k++){
			output.println(nbest.get(k).first);
			output.println(nbest.get(k).second);
			
			//get oracle
			double f1 = system.evaluateOneTree(testSents.get(j), nbest.get(k).second, goldTrees.get(j));
			if(f1 > bestF1 || bestTree == null){
				bestF1 = f1;
				bestTree = nbest.get(k).second;
			}
			
			if(f1 < worstF1 || worstTree == null){
				worstF1 = f1;
				worstTree = nbest.get(k).second;
			}
		}
		
		output.println();
		
		bestTrees.add(bestTree);
		worstTrees.add(worstTree);
		bestTree = null;
		worstTree = null;
		bestF1 = 0;
		worstF1 = 0;
	}
		
	
	
	output.close();
	
	System.err.println("Best Oracle");
	System.err.println(system.evaluate(testSents, bestTrees, goldTrees));
	
	System.err.println("Original Result");
	System.err.println(system.evaluate(testSents, originalTrees, goldTrees));
	
	System.err.println("Worst Oracle");
	System.err.println(system.evaluate(testSents, worstTrees, goldTrees));

}

 /**
   * Explicitly specifies the number of arguments expected with
   * particular command line options.
   */
  private static final Map<String, Integer> numArgs = new HashMap<>();
  static {
    numArgs.put("textFile", 1);
    numArgs.put("outFile", 1);
  }

  public static void main(String[] args) throws IOException {
    Properties props = StringUtils.argsToProperties(args, numArgs);
    DependencyParser parser = new DependencyParser(props);

    // Train with CoNLL-X data
    if (props.containsKey("trainFile")){
    	
    	if(props.containsKey("bUsePretraining") && props.containsKey("sPretrainingModel"))
    		parser.loadModelFile(props.getProperty("sPretrainingModel"));
      parser.train(props.getProperty("trainFile"), props.getProperty("devFile"), props.getProperty("model"),
          props.getProperty("embedFile"));
      }

    boolean loaded = false;
    
    // Test with CoNLL-X data
    if (props.containsKey("testFile")) {
      parser.loadModelFile(props.getProperty("model"));
      loaded = true;
      
      if(props.containsKey("bOutputReranking")){
    	  parser.getNBest(props.getProperty("testFile"), props.getProperty("outFile"));
      }
      else if(props.containsKey("bWrongActionNanlysis")){
    	  parser.outputWrongActProb(props.getProperty("testFile"), props.getProperty("outFile"));
      }
      //get selection braching oracle
      else if(props.containsKey("nOracleDepth")){
      	parser.loadModelFile(props.getProperty("model"));
      	parser.getOracle(props.getProperty("testFile"), props.getProperty("outFile"));
      }
      else{
    	  parser.testCoNLL(props.getProperty("testFile"), props.getProperty("outFile"));
      }
      
      //beam decoder with exitting model trained by greedy
//      if(props.containsKey("beamDecode")) 
//    	  parser.beamDecode(props.getProperty("testFile"), null);
//      else 
//    	  parser.testCoNLL(props.getProperty("testFile"), props.getProperty("outFile"));
    }
    
  

    // Parse raw text data
    if (props.containsKey("textFile")) {
      if (!loaded) {
        parser.loadModelFile(props.getProperty("model"));
        loaded = true;
      }

      String encoding = parser.config.tlp.getEncoding();
      String inputFilename = props.getProperty("textFile");
      BufferedReader input;
      try {
        input = inputFilename.equals("-")
                ? IOUtils.readerFromStdin(encoding)
                : IOUtils.readerFromString(inputFilename, encoding);
      } catch (IOException e) {
        throw new RuntimeIOException("No input file provided (use -textFile)", e);
      }

      String outputFilename = props.getProperty("outFile");
      PrintWriter output;
      try {
        output = outputFilename == null || outputFilename.equals("-")
            ? IOUtils.encodedOutputStreamPrintWriter(System.out, encoding, true)
            : IOUtils.getPrintWriter(outputFilename, encoding);
      } catch (IOException e) {
        throw new RuntimeIOException("Error opening output file", e);
      }

      parser.parseTextFile(input, output);
    }
  }

  /**
   * multi-beam decoding and return the best scored parse
   * 
   */
  private List<DependencyTree> predictInnerWithBeam(CoreMap sentence) {
	  
	  Triple<Double, Boolean, ActTypeBeam> predict = classifier.multiBeamDecoding(null, false, sentence, null);
	  List<DependencyTree> retval = new ArrayList<DependencyTree>();
	  // return the beam trees!
	  
	  for( HierarchicalDepState state : predict.third.returnBeamStates() )
		  retval.add(state.c.tree);
		  
	  return retval;
  }
  
  
private void beamDecode(String testFile, String outFile) {

	  System.err.println("Test File: " + testFile);
	  Timing timer = new Timing();
	  List<CoreMap> testSents = new ArrayList<>();
	  List<DependencyTree> testTrees = new ArrayList<DependencyTree>();
	  Util.loadConllFile(testFile, testSents, testTrees);
	  // count how much to parse
	  int numWords = 0;
	  int numSentences = 0;
	  for (CoreMap testSent : testSents) {
		  numSentences += 1;
		  numWords += testSent.get(CoreAnnotations.TokensAnnotation.class).size();
	  }
	  
	  List<List<DependencyTree>> predictedBeam = testSents.stream().map(this::predictInnerWithBeam).collect(toList());
	  system.evaluateOracle(testSents, predictedBeam, testTrees);
	  
	  List<DependencyTree> predicated = new ArrayList<DependencyTree>();
	  for(int i = 0; i<predictedBeam.size(); i++)
		  predicated.add(predictedBeam.get(i).get(0));
	  Map<String, Double>result = system.evaluate(testSents, predicated, testTrees);
	  double lasNoPunc = result.get("LASwoPunc");
	  System.err.printf("UAS = %.4f%n", result.get("UASwoPunc"));
	  System.err.printf("LAS = %.4f%n", lasNoPunc);
	  long millis = timer.stop();
	  double wordspersec = numWords / (((double) millis) / 1000);
	  double sentspersec = numSentences / (((double) millis) / 1000);
	  System.err.printf("%s tagged %d words in %d sentences in %.1fs at %.1f w/s, %.1f sent/s.%n",
			  StringUtils.getShortClassName(this), numWords, numSentences, millis / 1000.0, wordspersec, sentspersec);
	  
	  //if (outFile != null) {
		 // Util.writeConllFile(outFile, testSents, predicted);
	  

  }

}