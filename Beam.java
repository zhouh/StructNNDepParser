package nndep;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * 
 * @author Hao Zhou
 *
 */
public class Beam extends PriorityQueue<DepState>{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public int beam;
	
	public double maxScore = Double.NEGATIVE_INFINITY;
	public DepState maxScoreState = null;
	
	/**
	 * the constructor of the new priority queue
	 * 
	 */
	public Beam(int beam){
		
		//the new comparator function
		super(1,new Comparator<DepState>(){
			@Override
			public int compare(DepState o1, DepState o2) {
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
	public void insert(DepState item){
		if(item.score > maxScore){
			maxScoreState = item;
			maxScore = item.score;
		}
		
		if(this.size()<beam) {
			offer(item);
		}
		else if(item.score<=peek().score) return;
		else {
			poll();
			offer(item);
		}
	}
	
	public void clear() {
		this.clear();
	}
	
	public boolean beamGold(){
		for(DepState state : this){
			if(state.bGold)
				return true;
		}
		return false;
	}

	/**
	 * get the K largest item from the beam size min-heap
	 * 
	 */
//	public ChartItem[] getKBest(int k){
//		
//		//if the k is larger than beam
//		if(k>beam) throw new RuntimeException("The beam don't have enough K best item!");
//		
//		ChartItem[] items=new ChartItem[k];
//		
//		//insert the heap head items in the min-heap into a link list 
//		LinkedList<ChartItem> linkItems=new LinkedList<ChartItem>();
//		while(!this.isEmpty())
//			linkItems.push(this.poll());
//		
//		//get the last K insert item, the largest K items
//		for(int i=0;i<k;i++)
//			items[i]=linkItems.pop();
//		
//		return items;
//		
//	}

	
	

////    JUST FOR TEST
//	public static void main(String[] args){
//		ChartItem i0=new ChartItem(null, null,null, null, 1.0);
//		ChartItem i1=new ChartItem(null, null,null, null, 1.1);
//		ChartItem i2=new ChartItem(null, null, null, null,1.2);
//		
//		BeamChart chart=new BeamChart(3);
//		chart.insert(i1);
//		chart.insert(i0);
//		chart.insert(i2);
//		
//		System.out.println(chart.peebeam().getScore());
//		
//		for(ChartItem item:chart){
//			System.out.println(item.getScore());
//		}
//		
//		System.out.println("insert a new item");
//		chart.insert(new ChartItem(null, null, 1.6));
//		chart.insert(new ChartItem(null, null, 0.9));
//		chart.insert(new ChartItem(null, null, 1.8));
//		for(ChartItem item:chart){
//			System.out.println(item.getScore());
//		}
//	}
 
}  

