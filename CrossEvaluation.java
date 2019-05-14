

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

public class CrossEvaluation {
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
		System.out.println("=================================================");
		while (true) {
			fold = sc.nextInt();
			if (fold < 2) {
				System.out.println("ERROR: Please enter an integer greater than 2");
			}
			if (fold >= 2) {
				break;
			}
		}
		
        BufferedReader datafile = readDataFile("vote.arff");
        data = new Instances(datafile);
        
        data.setClassIndex(data.numAttributes() - 1);
        data.stratify(fold);
        Random rand = new Random(1);

        J48 clsJ48 = new J48();
        clsJ48.buildClassifier(data);
        Evaluation evalJ48 = new Evaluation(data);
        evalJ48.crossValidateModel(clsJ48, data, fold, rand);
        System.out.println(evalJ48.toSummaryString("==J48==", false));
        
        data.stratify(fold);
        JRip clsJRip = new JRip();
        clsJRip.buildClassifier(data);
        Evaluation evalJRip = new Evaluation(data);
        evalJRip.crossValidateModel(clsJRip, data, fold, rand);
        System.out.println(evalJRip.toSummaryString("==JRip==", false));
        
        data.stratify(fold);
        PART clsPART = new PART();
        clsPART.buildClassifier(data);
        Evaluation evalPART = new Evaluation(data);
        evalPART.crossValidateModel(clsPART, data, fold, rand);
        System.out.println(evalPART.toSummaryString("==PART==", false));
       
        
    }
}