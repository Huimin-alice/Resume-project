import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class CoOccurrenceMapper extends Mapper<Object, Text, Text, Text> {
	@Override
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		//input: movieB \t movieA=relation
		//do nothing, just pass data to reducer
		String[] line = value.toString().trim().split("\t");
		context.write(new Text(line[0]), new Text(line[1]));
	}
}
