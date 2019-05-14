import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class WekaTest2 {
	public static String algorithm;
	public static String dataString;
	public static int folds;
	public static Instances data;
	
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
	
	
	
	
	public static void main(String[] args) throws IOException {
		Scanner sc = new Scanner(System.in);
		
		System.out.println("Enter value k for number of folds: ");
		System.out.println("(Entering '5' will give you 80% Training and 20% Testing.)");
		System.out.println("(Entering '10' will give you 90% Training and 10% Testing.)");
		while (true) {
			folds = sc.nextInt();
			if (folds <= 2) {
				System.out.println("ERROR: Please enter an integer greater than 2");
			}
			if (folds > 2) {
				break;
			}
		}
		BufferedReader datafile = readDataFile("iris.arff");
        data = new Instances(datafile);
        
        data.setClassIndex(data.numAttributes() - 1);
        
        
        
        Classifier cls = new J48();
        Evaluation eval = new Evaluation(data);
        Random rand = new Random(1);
        int folds = 10;
        eval.crossValidateModel(cls, data, folds, rand);
        System.out.println(eval.toSummaryString());
		
		
	}



	
}
