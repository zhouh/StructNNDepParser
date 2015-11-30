package nndep;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;

/**
 * 
 * @author Hao Zhou
 *
 */
public class ActTypeBeam extends PriorityBlockingQueue<DepTypeBeam>{
	
private static final long serialVersionUID = 1L;
	
	public int beam;
	
	public double maxScore = Double.NEGATIVE_INFINITY;
	public HierarchicalDepState maxScoreState = null;
	public double bestDepTypeScore = Double.NEGATIVE_INFINITY;
	public double bestActTypeScore = Double.NEGATIVE_INFINITY;
	
	/**
	 * the constructor of the new priority queue
	 * 
	 */
	public ActTypeBeam(int beam){
		
		//the new comparator function
		super(1,new Comparator<DepTypeBeam>(){
			@Override
			public int compare(DepTypeBeam o1, DepTypeBeam o2) {
				if( o1.score < o2.score)
					return -1 ;
				else if(o1.score == o2.score){
					return 0;
				}else{
					return 1;
				}
			}   	 
	     });
		
		this.beam=beam;
		
	}
	
	/**
	 * insert a new item into the chart
	 * the chart always keep the best beam item in the chart
	 * @param item
	 */
	public void insert(DepTypeBeam item){
		
		if(this.size()<beam) {
			offer(item);
		}
		else if(item.score<=peek().score) return;
		else {
			poll();
			offer(item);
		}
	}
	
	public void clearAll() {
		this.clear();
	}
	
	public boolean containGold(){
		for(DepTypeBeam b : this){
			if(b.containGold())
				return true;
		}
		return false;
	}
	
	public HierarchicalDepState getBestState(){
		HierarchicalDepState bestState = null;
		double bestScore = Double.NEGATIVE_INFINITY;
		
		for(DepTypeBeam dtBeam : this)
			for(HierarchicalDepState state : dtBeam){
				if(state.score > bestScore){
					bestState = state;
					bestScore = state.score;
				}
				if(state.actTypeScore > bestActTypeScore)
					bestActTypeScore = state.actTypeScore;
				if(state.depTypeScore > bestDepTypeScore)
					bestDepTypeScore = state.depTypeScore;
			}
		
			
		return bestState;
	}

	public List<HierarchicalDepState> returnBeamStates() {

		List<HierarchicalDepState> retval = new ArrayList<>();
		for(DepTypeBeam dtBeam : this)
			for( HierarchicalDepState state : dtBeam )
				retval.add(state);
		return retval;
		
	}
	
	public void display(){
		int i = 0;
		int j = 0;
		System.out.println("======================");
		for(DepTypeBeam dtBeam : this){
			System.out.println(i++ +" :");
			j = 0;
			System.out.println("-----");
			for(HierarchicalDepState state : dtBeam){
				System.out.println((i-1) +"-"+ j++ + " "+state.bGold);
				System.out.println(state.actionSequence());
				System.out.println(state.score + "\t"+ state.actTypeScore + "\t"+state.depTypeScore);
			}
			
		}
	}
	
}  

