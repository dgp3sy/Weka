

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import javax.activation.DataSource;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class EvaluateOptions {
    public static Instances data;
    public static int fold;
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
	
    
    public static void main(String[] args) throws Exception {
        
    	Scanner sc = new Scanner(System.in);
		
		System.out.println("Enter value k for number of folds: ");
		System.out.println("(Entering '5' will give you 80% Training and 20% Testing.)");
		System.out.println("(Entering '10' will give you 90% Training and 10% Testing.)");
		while (true) {
			fold = sc.nextInt();
			if (fold <= 2) {
				System.out.println("ERROR: Please enter an integer greater than 2");
			}
			if (fold > 2) {
				break;
			}
		}
		
        BufferedReader datafile = readDataFile("iris.arff");
        data = new Instances(datafile);
        
        data.setClassIndex(data.numAttributes() - 1);
        
        String[] options = new String[1];
        options[0] = "-C 0.25-M 2";
        J48 tree = new J48();
        tree.setOptions(options);
        tree.buildClassifier(data);
        
        System.out.println();
        
       
        
    }
}