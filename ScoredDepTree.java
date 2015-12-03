package nndep;

public class ScoredDepTree  implements Comparable{

	public DependencyTree tree;
	public double score;
	
	public ScoredDepTree(DependencyTree tree, double score) {
		super();
		this.tree = tree;
		this.score = score;
	}
	
	@Override
	public int compareTo(Object o) {

		ScoredDepTree s = (ScoredDepTree)o;
		int retval = score > s.score ? -1 : (score == s.score ? 0 : 1);
		return retval;
	}
}
