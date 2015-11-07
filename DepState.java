package nndep;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class DepState implements Comparable{
	
	public double score;
	public Configuration c;
	public int act;
	public DepState lastState = null;
	public boolean bGold = true;
	public List<Integer> labels = null;
	int[] dropOutArray = null;
	int[] featureArray = null;
	
	public DepState(Configuration c, Integer act, double score) {
		
		this.c = c;
		this.act = act;
		this.score = score;
		
	}
	
	public DepState(Configuration c, Integer act, double score, DepState last, boolean bGold) {
		
		this.c = c;
		this.act = act;
		this.score = score;
		this.lastState = last;
		this.bGold = bGold;
		
	}
	
	public void setLabel(List<Integer> label){
		this.labels = label;
		
	}
	
	public void setDropOutArray(int[] dropOutArray){
		this.dropOutArray = dropOutArray;
	}
	
	public void setFeatureArray(int[] featureArray){
		this.featureArray = featureArray;
	}
	
	public void StateApply(ParsingSystem system){
		
		c = new Configuration(c);
		system.apply(c, act);
	}
	
	public List<Integer> actionSequence(){
		
		ArrayList<Integer> retval = new ArrayList<Integer>();
		DepState statePtr = this;
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
