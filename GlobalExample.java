package nndep;

import java.util.List;

import edu.stanford.nlp.util.CoreMap;

public class GlobalExample {

	public  CoreMap sent;
	public DependencyTree oracle;
	public List<Example> examples;
	public List<Integer> acts;
	
	public GlobalExample(CoreMap sent, DependencyTree oracle, List<Example> examples, List<Integer> acts){
		this.sent = sent;
		this.oracle = oracle;
		this.examples = examples;
		this.acts = acts;
		
	}
	
	public List<Example> getExamples(){
		return examples;
	}
	
	
}
