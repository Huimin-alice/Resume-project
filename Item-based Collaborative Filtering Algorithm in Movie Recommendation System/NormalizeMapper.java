import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class NormalizeMapper extends Mapper<Object, Text, Text, Text> {
	@Override
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

		//movieA:movieB \t relation
		//collect the relationship list for movieA
		String[] movie_relation = value.toString().trim().split("\t");
		String[] movies = movie_relation[0].split(":");
		StringBuilder builder = new StringBuilder();
		builder.append(movies[1]).append(":").append(movie_relation[1]);
		context.write(new Text(movies[0]), new Text(builder.toString()));
	}
}