package nndep;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.stanford.nlp.util.Pair;

public class HierarchicalDepState implements Comparable{

	public double score;
	public double actTypeScore;
	public double depTypeScore;
	public Configuration c;
	public int actType;
	public int depType;
	public HierarchicalDepState lastState = null;
	public boolean bGold = true;
	public double[] actTypeDistribution = null;
	public int[] actTypeLabel = null;
	public int[] depTypeLabel = null;
	int[] featureArray = null;
	HiddenLayer hiddenLayer = null;
	
	public HierarchicalDepState(Configuration c, int actType,
			int depType, double actTypeScore, double depTypeScore, HierarchicalDepState last, boolean bGold) {
		
		this.c = c;
		this.actType = actType;
		this.depType = depType;
		this.actTypeScore = actTypeScore;
		this.depTypeScore = depTypeScore;
		this.lastState = last;
		this.bGold = bGold;
		this.score = this.actTypeScore + this.depTypeScore; // when generate the state, the score is the action type score
	}
	
	public HierarchicalDepState(double score, double actTypeScore, double depTypeScore, Configuration c, int actType,
			int depType, HierarchicalDepState lastState, boolean bGold, int[] actTypeLabel, int[] depTypeLabel,
			int[] featureArray, HiddenLayer hiddenLayer) {

		this.actTypeScore = actTypeScore;
		this.depTypeScore = depTypeScore;
		this.c = c;
		this.actType = actType;
		this.depType = depType;
		this.lastState = lastState;
		this.bGold = bGold;
		this.actTypeLabel = actTypeLabel;
		this.depTypeLabel = depTypeLabel;
		this.featureArray = featureArray;
		this.hiddenLayer = hiddenLayer;
		this.score = depTypeScore + actTypeScore;
	}


	public void setActTypeDistribution(double[] actTypeDistribution) {
		this.actTypeDistribution = actTypeDistribution;
	}


	public void setActTypeLabel(int[] actTypeLabel) {
		this.actTypeLabel = actTypeLabel;
	}

	public void setHiddenLayer(HiddenLayer hiddenLayer) {
		this.hiddenLayer = hiddenLayer;
	}

	public void setFeatureArray(int[] featureArray){
		this.featureArray = featureArray;
	}
	
	public HierarchicalDepState getNewStateForExpandDepType(int depType2, double depTypeScore2, 
			boolean bGold, int[] depTypeLabel2){
		
		HierarchicalDepState state =  new HierarchicalDepState(score, actTypeScore, 
				depTypeScore2, c, actType,
			depType2, lastState, bGold, actTypeLabel, depTypeLabel2,
			featureArray, hiddenLayer);
		return state;
		
	}
	
	public void StateApply(ParsingSystem system){
		
		c = new Configuration(c);
		system.apply(c, actType, depType);
	}
	
	public List<Pair<Integer, Integer>> actionSequence(){
		
		ArrayList<Pair<Integer, Integer>> retval = new ArrayList<Pair<Integer, Integer>>();
		HierarchicalDepState statePtr = this;
		while(statePtr.actType != -1){
			retval.add( new Pair<Integer, Integer>(statePtr.actType, statePtr.depType) );
			statePtr = statePtr.lastState;
		}
		
		 Collections.reverse(retval);
		 
		 return retval;
			
	}

	/**
	 *   For sort from large to small
	 */
	@Override
	public int compareTo(Object o) {

		HierarchicalDepState s = (HierarchicalDepState)o;
		int retval = score > s.score ? -1 : (score == s.score ? 0 : 1);
		return retval;
	}

	@Override
	public String toString() {
		return "DepState [score=" + score + ", c=" + c + ", bGold=" + bGold
				+ "]";
	}

	public void setParas(HiddenLayer hiddenLayer2, int[] actTypeLabel2, int[] featureArray2) {

		this.hiddenLayer = hiddenLayer2;
		this.actTypeLabel = actTypeLabel2;
		this.featureArray = featureArray2;
	}


}
