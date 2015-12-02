package nndep;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class HierarchicalDepState implements Comparable{

	public double score;
	public Configuration c;
	public int act;
	public HierarchicalDepState lastState = null;
	public boolean bGold = true;
	public List<Integer> labels = null;
	HiddenLayer hiddenLayer = null;
	int[] featureArray = null;
	
	public HierarchicalDepState(Configuration c, int act, double score, HierarchicalDepState last, boolean bGold) {
		
		this.c = c;
		this.act = act;
		this.score = score;
		this.lastState = last;
		this.bGold = bGold;
		
	}
	
	public void setLabel(List<Integer> label){
		this.labels = label;
		
	}
	
	public void setFeatureArray(int[] featureArray){
		this.featureArray = featureArray;
	}
	
	public void setHidden(HiddenLayer hiddenLayer2) {

		this.hiddenLayer = hiddenLayer2;
	}
	
	public void StateApply(ParsingSystem system){
		
		c = new Configuration(c);
		system.apply(c, act);
	}
	
	public List<Integer> actionSequence(){
		
		ArrayList<Integer> retval = new ArrayList<Integer>();
		HierarchicalDepState statePtr = this;
		while(statePtr.act != -1){
			retval.add(statePtr.act);
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

		DepState s = (DepState)o;
		int retval = score > s.score ? -1 : (score == s.score ? 0 : 1);
		return retval;
	}

	@Override
	public String toString() {
		return "DepState [action sequence: "+actionSequence()+ " score=" + score + ", c=" + c + ", bGold=" + bGold
				+ "]";
	}



}