

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import javax.activation.DataSource;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Evaluate {
	
	private static Instances[][] crossValidationSplit(Instances data2, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
		
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data2.trainCV(numberOfFolds, i);
			split[1][i] = data2.testCV(numberOfFolds, i);
		}
		return split;
	}
	
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
        Instances data;
        double acJ48;
        int trainSize, testSize;
        Instances train, test;
        
        BufferedReader datafile = readDataFile("iris.arff");
        data = new Instances(datafile);
        
        data.setClassIndex(data.numAttributes() - 1);
        
        data.randomize(new Random(1));
        trainSize = (int) Math.round(data.numInstances() * 0.66);
        testSize = (int) Math.round(data.numInstances() - trainSize);
        train = new Instances(data, 0, trainSize);
        test = new Instances(data, trainSize, testSize);
        
        
        J48 tree = new J48();
        tree.buildClassifier(train);
        
        Evaluation eval = new Evaluation(data);
        eval = new Evaluation(train);
        eval.evaluateModel(tree, test);
        acJ48 = eval.pctCorrect();
        System.out.println(acJ48);
        System.out.println(eval.toSummaryString("==J48==", false));
        
    }
}