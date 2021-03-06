package nndep;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.CoreMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Defines a transition-based parsing framework for dependency parsing.
 *
 * @author Danqi Chen
 */
public abstract class ParsingSystem {

  /**
   * Defines language-specific settings for this parsing instance.
   */
  private final TreebankLanguagePack tlp;

  /**
   * Dependency label used between root of sentence and ROOT node
   */
  protected final String rootLabel;

  protected List<String> labels, transitions;

  /**
   * Generate all possible transitions which this parsing system can
   * take for any given configuration.
   */
  protected abstract void makeTransitions();

  /**
   * Determine whether the given transition is legal for this
   * configuration.
   *
   * @param c Parsing configuration
   * @param t Transition string
   * @return Whether the given transition is legal in this
   *         configuration
   */
  public abstract boolean canApply(Configuration c, String t);

  /**
   * Apply the given transition to the given configuration, modifying
   * the configuration's state in place.
   */
  public abstract void apply(Configuration c, String t);
  
  public void apply(Configuration c, int t){
	  apply(c, transitions.get(t));
  }
  
  /*
   * hierarchical dep action paras
   */
  public int nActTypeNum, nDepTypeNum, nShiftID, nLeftID, nRightID;
  
  public abstract int getHierarchicalActID(int actType, int depType);
	  

  /**
   * Provide a static-oracle recommendation for the next parsing step
   * to take.
   *
   * @param c Current parser configuration
   * @param dTree Gold tree which parser needs to reach
   * @return Transition string
   */
  public abstract String getOracle(Configuration c, DependencyTree dTree);

  /**
   * Determine whether applying the given transition in the given
   * configuration tree will leave in us a state in which we can reach
   * the gold tree. (Useful for building a dynamic oracle.)
   */
  abstract boolean isOracle(Configuration c, String t, DependencyTree dTree);

  /**
   * Build an initial parser configuration from the given sentence.
   */
  public abstract Configuration initialConfiguration(CoreMap sentence);

  /**
   * Determine if the given configuration corresponds to a parser which
   * has completed its parse.
   */
  abstract boolean isTerminal(Configuration c);

  // TODO pass labels as Map<String, GrammaticalRelation>; use
  // GrammaticalRelation throughout

  /**
   * @param tlp TreebankLanguagePack describing the language being
   *            parsed
   * @param labels A list of possible dependency relation labels, with
   *               the ROOT relation label as the first element
   */
  public ParsingSystem(TreebankLanguagePack tlp, List<String> labels, boolean verbose) {
    this.tlp = tlp;
    this.labels = new ArrayList<>(labels);

    //NOTE: assume that the first element of labels is rootLabel
    rootLabel = labels.get(0);
    makeTransitions();

    if (verbose) {
      System.err.println(Config.SEPARATOR);
      System.err.println("#Transitions: " + transitions.size());
      System.err.println("#Labels: " + labels.size());
      System.err.println("ROOTLABEL: " + rootLabel);
    }
  }

  public int getTransitionID(String s) {
    for (int k = 0; k < transitions.size(); ++k)
      if (transitions.get(k).equals(s))
        return k;
    return -1;
  }

  private Set<String> getPunctuationTags() {
    if (tlp instanceof PennTreebankLanguagePack) {
      // Hack for English: match punctuation tags used in Danqi's paper
      return new HashSet<>(Arrays.asList("``", "''", ".", ",", ":"));
    } else {
      return CollectionUtils.asSet(tlp.punctuationTags());
    }
  }

  /**
   * Evaluate performance on a list of sentences, predicted parses,
   * and gold parses.
   *
   * @return A map from metric name to metric value
   */
  public Map<String, Double> evaluate(List<CoreMap> sentences, List<DependencyTree> trees,
                                      List<DependencyTree> goldTrees) {
    Map<String, Double> result = new HashMap<String, Double>();

    // We'll skip words which are punctuation. Retrieve tags indicating
    // punctuation in this treebank.
    Set<String> punctuationTags = getPunctuationTags();

    if (trees.size() != goldTrees.size()) {
      System.err.println("[Error] Incorrect number of trees.");
      return null;
    }

    int correctArcs = 0;
    int correctArcsWoPunc = 0;
    int correctHeads = 0;
    int correctHeadsWoPunc = 0;

    int correctTrees = 0;
    int correctTreesWoPunc = 0;
    int correctRoot = 0;

    int sumArcs = 0;
    int sumArcsWoPunc = 0;

    for (int i = 0; i < trees.size(); ++i) {
      List<CoreLabel> tokens = sentences.get(i).get(CoreAnnotations.TokensAnnotation.class);

      if (trees.get(i).n != goldTrees.get(i).n) {
        System.err.println("[Error] Tree " + (i + 1) + ": incorrect number of nodes.");
        return null;
      }
      if (!trees.get(i).isTree()) {
        System.err.println("[Error] Tree " + (i + 1) + ": illegal.");
        return null;
      }

      int nCorrectHead = 0;
      int nCorrectHeadwoPunc = 0;
      int nonPunc = 0;

      for (int j = 1; j <= trees.get(i).n; ++j) {
        if (trees.get(i).getHead(j) == goldTrees.get(i).getHead(j)) {
          ++correctHeads;
          ++nCorrectHead;
          if (trees.get(i).getLabel(j).equals(goldTrees.get(i).getLabel(j)))
            ++correctArcs;
        }
        ++sumArcs;

        String tag = tokens.get(j - 1).tag();
        if (!punctuationTags.contains(tag)) {
          ++sumArcsWoPunc;
          ++nonPunc;
          if (trees.get(i).getHead(j) == goldTrees.get(i).getHead(j)) {
            ++correctHeadsWoPunc;
            ++nCorrectHeadwoPunc;
            if (trees.get(i).getLabel(j).equals(goldTrees.get(i).getLabel(j)))
              ++correctArcsWoPunc;
          }
        }
      }
      if (nCorrectHead == trees.get(i).n)
        ++correctTrees;
      if (nCorrectHeadwoPunc == nonPunc)
        ++correctTreesWoPunc;
      if (trees.get(i).getRoot() == goldTrees.get(i).getRoot())
        ++correctRoot;
    }

    result.put("UAS", correctHeads * 100.0 / sumArcs);
    result.put("UASwoPunc", correctHeadsWoPunc * 100.0 / sumArcsWoPunc);
    result.put("LAS", correctArcs * 100.0 / sumArcs);
    result.put("LASwoPunc", correctArcsWoPunc * 100.0 / sumArcsWoPunc);

    result.put("UEM", correctTrees * 100.0 / trees.size());
    result.put("UEMwoPunc", correctTreesWoPunc * 100.0 / trees.size());
    result.put("ROOT", correctRoot * 100.0 / trees.size());


    return result;
  }
  
  /**
   * Evaluate oracle on a list of sentences, k-best  predicted parses,
   * and gold parses.
   *
   * @return A map from metric name to metric value
   */
  public void evaluateOracle(List<CoreMap> sentences, List<List<DependencyTree>> trees,
                                      List<DependencyTree> goldTrees) {
    Map<String, Double> result = new HashMap<String, Double>();

    // We'll skip words which are punctuation. Retrieve tags indicating
    // punctuation in this treebank.
    Set<String> punctuationTags = getPunctuationTags();

    if (trees.size() != goldTrees.size()) {
      System.err.println("[Error] Incorrect number of trees.");
      return ;
    }
    
    double correct = 0;
    double sum = 0;
    
    for(int i = 0; i<sentences.size(); i++){
    	Map<String, Double> oneSentResult = evaluateOnrSent(sentences.get(i), trees.get(i), goldTrees.get(i));
    	correct += oneSentResult.get("correct");
    	sum += oneSentResult.get("sum");
    }
    
    System.out.println("UAS w/o puntuations of oracles: " + correct/sum);
    
   
  }
  
  /**
   *  
   *  get the UAS of a sentence other than a corpus
   * 
   */
  public Map<String, Double> evaluateOnrSent(CoreMap sentence,List<DependencyTree> trees,
          DependencyTree goldTree) {
  
	  Map<String, Double> result = new HashMap<String, Double>();
	  Set<String> punctuationTags = getPunctuationTags();
	  
	  //get the best tree of a sentence from beams and return the result of them
	  double bestUas = 0;
	  double best_correctTreesWoPunc = 0;
	  double best_sumArcsWoPunc = 0;

	    for (int i = 0; i < trees.size(); ++i) {
	    	
	    	  int correctArcs = 0;
	  	    int correctArcsWoPunc = 0;
	  	    int correctHeads = 0;
	  	    int correctHeadsWoPunc = 0;

	  	    int correctTrees = 0;
	  	    int correctTreesWoPunc = 0;
	  	    int correctRoot = 0;

	  	    int sumArcs = 0;
	  	    int sumArcsWoPunc = 0;
	  	    
	      List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

	      if (trees.get(i).n != goldTree.n) {
	        System.err.println("[Error] Tree " + (i + 1) + ": incorrect number of nodes.");
	        return null;
	      }
	      if (!trees.get(i).isTree()) {
	        System.err.println("[Error] Tree " + (i + 1) + ": illegal.");
	        return null;
	      }

	      int nCorrectHead = 0;
	      int nCorrectHeadwoPunc = 0;
	      int nonPunc = 0;

	      for (int j = 1; j <= trees.get(i).n; ++j) {
	        if (trees.get(i).getHead(j) == goldTree.getHead(j)) {
	          ++correctHeads;
	          ++nCorrectHead;
	          if (trees.get(i).getLabel(j).equals(goldTree.getLabel(j)))
	            ++correctArcs;
	        }
	        ++sumArcs;

	        String tag = tokens.get(j - 1).tag();
	        if (!punctuationTags.contains(tag)) {
	          ++sumArcsWoPunc;
	          ++nonPunc;
	          if (trees.get(i).getHead(j) == goldTree.getHead(j)) {
	            ++correctHeadsWoPunc;
	            ++nCorrectHeadwoPunc;
	            if (trees.get(i).getLabel(j).equals(goldTree.getLabel(j)))
	              ++correctArcsWoPunc;
	          }
	        }
	      }
	      if (nCorrectHead == trees.get(i).n)
	        ++correctTrees;
	      if (nCorrectHeadwoPunc == nonPunc)
	        ++correctTreesWoPunc;
	      if (trees.get(i).getRoot() == goldTree.getRoot())
	        ++correctRoot;

	      // compute the result!
	      double uas = correctHeadsWoPunc * 100.0 / sumArcsWoPunc;
	      
	      if(uas > bestUas){
	    	  bestUas = uas;
	    	  best_correctTreesWoPunc = correctHeadsWoPunc;
	    	  best_sumArcsWoPunc = sumArcsWoPunc;
	      }
	    }
	    result.put("correct", best_correctTreesWoPunc);
	    result.put("sum", best_sumArcsWoPunc);


	    return result;
  }

  public double getUASScore(List<CoreMap> sentences, List<DependencyTree> trees, List<DependencyTree> goldTrees) {
    Map<String, Double> result = evaluate(sentences, trees, goldTrees);
    return result == null || !result.containsKey("UASwoPunc") ? -1.0 : result.get("UASwoPunc");
  }

}
