package nndep;

public class HiddenLayer {
	
	double[] hidden;
	double[] hidden3;
	int[] dropOut;
	
	public HiddenLayer(double[] hidden, double[] hidden3, int[] dropOut) {
		super();
		this.hidden = hidden;
		this.hidden3 = hidden3;
		this.dropOut = dropOut;
	}
	
}