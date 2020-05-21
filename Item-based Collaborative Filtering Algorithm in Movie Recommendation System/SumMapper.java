import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class SumMapper extends Mapper<Object, Text, Text, DoubleWritable> {

	@Override
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

		//pass data to reducer
		String[] line = value.toString().split("\t");
		context.write(new Text(line[0]), new DoubleWritable(Double.parseDouble(line[1])));
	}
}
