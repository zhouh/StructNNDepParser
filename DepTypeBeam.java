package nndep;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;

public class DepTypeBeam extends PriorityBlockingQueue<HierarchicalDepState>{

	private static final long serialVersionUID = 1L;

	public int beam;
	
	public double score = Double.NEGATIVE_INFINITY;
	
	public List<HierarchicalDepState> bucket;
	
	/**
	 * the constructor of the new priority queue
	 * 
	 */
	public DepTypeBeam(int beam){
		
		//the new comparator function
		super(1,new Comparator<HierarchicalDepState>(){
			@Override
			public int compare(HierarchicalDepState o1, HierarchicalDepState o2) {
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
		this.bucket = new ArrayList<>();
		
	}
	
	/**
	 * add the state into bucket
	 * @param item
	 */
	public void insertBucket(HierarchicalDepState item){
		if(item.score > score)
			score = item.score;
		
		bucket.add(item);
		
	}
	
//	public void prune(){
//		for(HierarchicalDepState state : bucket)
//			insert(state);
//		
//		bucket.clear();
//	}
	
	/**
	 * insert a new item into the chart
	 * the chart always keep the best beam item in the chart
	 * @param item
	 */
	public void insert(HierarchicalDepState item){
		
		if(this.size()<beam) {
			offer(item);
		}
		else if(item.score<=peek().score) return;
		else {
			poll();
			offer(item);
		}
	}
	
	public void bucketClear(){
		bucket.clear();
	}
	
	public void clearAll() {
		this.clear();
	}
	
	public boolean containGold(){
		for(HierarchicalDepState state : this){
			if(state.bGold)
				return true;
		}
		return false;
	}
}
