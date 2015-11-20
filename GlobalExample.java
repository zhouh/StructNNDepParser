package nndep;

import java.util.List;

import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class GlobalExample {

	public  CoreMap sent;
	public DependencyTree oracle;
	public List<Example> examples;
	public List<Pair<Integer, Integer>> oracles;
	
	public GlobalExample(CoreMap sent, DependencyTree oracle, List<Example> examples, 
			List<Pair<Integer, Integer>> oracles){
		this.sent = sent;
		this.oracle = oracle;
		this.examples = examples;
		this.oracles = oracles;
		
	}
	
	public List<Example> getExamples(){
		return examples;
	}
	
	
}
