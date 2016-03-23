package nndep;


public class ReviseItem implements Comparable{

	public int stateIndex;
	public int reviseActID;
	double margin;
	
	public ReviseItem(int stateIndex, int reviseActID, double margin) {
		this.stateIndex = stateIndex;
		this.reviseActID = reviseActID;
		this.margin = margin;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		ReviseItem that = (ReviseItem) o;

		if (stateIndex != that.stateIndex) return false;
		return reviseActID == that.reviseActID;

	}

	@Override
	public int hashCode() {
		int result = stateIndex;
		result = 31 * result + reviseActID;
		return result;
	}

	/**
	 *   For sort from large to small
	 */
	@Override
	public int compareTo(Object o) {

		ReviseItem s = (ReviseItem)o;
		int retval = margin > s.margin ? 1 : (margin == s.margin ? 0 : -1);
		return retval;
	}

}
