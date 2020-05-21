import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class SumReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
	@Override
	public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
			throws IOException, InterruptedException {

		//user:movie relation
		//calculate the sum
		double sum = 0;
		for (DoubleWritable value : values){
			sum += value.get();
		}
		context.write(key, new DoubleWritable(sum));
	}
}
