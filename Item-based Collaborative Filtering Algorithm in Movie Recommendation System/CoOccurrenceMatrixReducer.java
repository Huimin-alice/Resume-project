import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class CoOccurrenceMatrixReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
	@Override
	public void reduce(Text key, Iterable<IntWritable> values, Context context)
			throws IOException, InterruptedException {
		//key movie1:movie2 value = iterable<1, 1, 1>
		//calculate each two movies have been watched by how many people
		int sum = 0;
		for (IntWritable value : values) {
			sum += value.get();
		}
		context.write(key, new IntWritable(sum));
	}
}