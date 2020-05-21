import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MultiplicationReducer extends Reducer<Text, Text, Text, DoubleWritable> {
	@Override
	public void reduce(Text key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {

		//key = movieB
		// value = <movieA=relation, movieC=relation... userA:rating, userB:rating...>
		//collect the data for each movie, then do the multiplication
		Map<String, Double> movieRelation = new HashMap<String, Double>();
		Map<String, Double> userRating = new HashMap<String, Double>();
		for (Text value : values) {
			String val = value.toString().trim();
			if (val.contains("=")){
				String[] movie_relation = val.split("=");
				movieRelation.put(movie_relation[0], Double.parseDouble(movie_relation[1]));
			} else if (val.contains(":")){
				String[] user_rating = val.split(":");
				userRating.put(user_rating[0], Double.parseDouble(user_rating[1]));
			}
		}

		StringBuilder builder = new StringBuilder();
		for (Map.Entry<String, Double> movie_relation : movieRelation.entrySet()) {
			String movieA = movie_relation.getKey();
			double relation = movie_relation.getValue();
			for (Map.Entry<String, Double> user_rating : userRating.entrySet()) {
				String user = user_rating.getKey();
				double rating = user_rating.getValue();
				builder.append(user).append(":").append(movieA);
				context.write(new Text(builder.toString()), new DoubleWritable(relation * rating));
				builder.setLength(0);
			}
		}
	}
}