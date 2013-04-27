package at.ac.tuwien.ifs.ir;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;
import weka.core.converters.TextDirectoryLoader;
import weka.core.tokenizers.WordTokenizer;
import at.ac.tuwien.ifs.ir.TextRetrieval.postingListSize;

public class OkapiBM25 {

    private static Logger log = LoggerFactory.getLogger(OkapiBM25.class);

    private String indexFile = null;
    private String source = ".";
    private String target = ".";
    private postingListSize postingListSize;
    
    // document length of the documents in the collection
    private Map<String, Integer> Ds;
    // average length of the documents in the collection
    private double avgdl;
    
    // store for the computed IDFs of the documents terms
    private Map<String,Double> idfs;

    private double k1 = 1.2;
    private double b = 0.75;

    public OkapiBM25(String indexFile, String source, String target, TextRetrieval.postingListSize postingListSize) {
        this.indexFile = indexFile;
        this.source = source;
        this.target = target;
        this.postingListSize = postingListSize; 
    }
    
    public void findSimilar(String[] documentIDs) {        
        log.info("Started Okapi BM25 retrieval ...");
        computeGlobals();
        for (int i = 0; i < documentIDs.length; i++)
            findSimilar(documentIDs[i], i + 1);
        log.info("Done Okapi BM25 retrieval");
    }
    
    public void findSimilar(String documentID, int topicNumber) {
        log.info("Started searching for similar documents for " + documentID + " ...");

        Instances index = null;
        try {
            index = new DataSource(indexFile).getDataSet();

            int queryIndex;
            boolean found = false;
            for (queryIndex = 0; queryIndex < index.numInstances(); queryIndex++) {
                if ((index.instance(queryIndex).stringValue(0)).equals(documentID)) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                log.warn("Requested query document with documentID of " + documentID + " could not be found");
                return;
            }
            
            Rank[] top10Ranks = new Rank[10];
            for (int i = 0; i < 10; i++)
                top10Ranks[i] = new Rank(0.0, null);

            idfs = new HashMap<String,Double>();
            for (int i = 0; i < index.numInstances(); i++) {
                if (i == queryIndex)
                    continue;

                double newSimilarity = computeBM25Score(index, queryIndex, i);
                
                if (top10Ranks[0].score < newSimilarity)
                    top10Ranks[0] = new Rank(newSimilarity, index.instance(i).stringValue(0));
                Arrays.sort(top10Ranks);
            }
            log.info("Found 10 most similar documents for " + documentID);

            String filename = target + "/" + postingListSize + "_topic" + topicNumber + "_groupG.txt";
            try {
                File file = new File(filename.substring(0, filename.lastIndexOf("/")));
                if (!file.exists()) { 
                    file.mkdirs();
                }
                file = new File(filename);
                if (!file.exists()) { 
                    file.createNewFile();
                }
                
                FileWriter fileWriter = new FileWriter(filename);
                BufferedWriter out = new BufferedWriter(fileWriter);
                Collections.reverse(Arrays.asList(top10Ranks));
                for (int i = 0; i < 10; i++) {
                    out.write("topic" + topicNumber
                            + " Q0 " + top10Ranks[i].documentID 
                            + " " + (i + 1)
                            + " " + top10Ranks[i].score
                            + " groupG_" + postingListSize + "\r\n");
                }
                out.close();
                
                log.info("Wrote results to " + filename);
            } catch (IOException ioe) {
                log.error("Error saving results to file " + filename, ioe);
                return;
            }
            log.info("Done");

        } catch (Exception e) {
            log.error("Error computing cosine similarities", e);
            return;
        }
    }

    private void computeGlobals() {
        log.info("Started computing document lengths ...");

        Instances instances = null;
        try {
            log.info("Loading files ...");
            TextDirectoryLoader tdl = new TextDirectoryLoader();
            tdl.setDirectory(new File(source));
            tdl.setRetrieval(Loader.INCREMENTAL);
            tdl.setOutputFilename(true);
            instances  = tdl.getDataSet();
        } catch (IOException ioe) {
            log.error("Error loading files", ioe);
            return;
        }

        log.info("Processing contents ...");
        // we implement the dictionary as a hash map, alternative would be tree map
        Ds = new HashMap<String,Integer>();
        WordTokenizer wordTokenizer = new WordTokenizer();
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        
        for (int i = 0; i < instances.numInstances(); i++) {
            int count = 0;
            wordTokenizer.tokenize(instances.instance(i).stringValue(0));
            while (wordTokenizer.hasMoreElements()) {
                wordTokenizer.nextElement();
                count++;
            }
            Ds.put(instances.instance(i).stringValue(1), count);
        }
        
        // compute average document length of the collection
        avgdl = 0.0;
        for (Iterator<String> i = Ds.keySet().iterator(); i.hasNext(); ) {
            avgdl += (double) Ds.get(i.next());
        }
        avgdl = avgdl / (double) Ds.size();
        log.info("Computed average document length of the collection as " + avgdl);
        log.info("Done computing document lengths");
    }

    private double computeBM25Score(Instances index, int queryIndex, int documentIndex) {

        double score = 0.0;
        
        for (int i = 2; i < index.numAttributes(); i++ ) {
            
            double queryValue = index.instance(queryIndex).value(i);
            // we skip the computation of terms that are not contained in the query
            if (Double.isNaN(queryValue) || queryValue <= 0.0)
                continue;

            double termFrequency = index.instance(documentIndex).value(i);
            termFrequency = Double.isNaN(termFrequency) ? 0 : termFrequency;

            String termName = index.attribute(i).name();
            double idf;
            if (idfs.containsKey(termName)) {
                idf = idfs.get(termName);
            } else {
                // compute the IDF for the current term
                // we use IDF(qi) = log ( N - n(qi) + 0.5 / n(qi) + 0.5)
                int containingTerm = 1;
                for (int j = 0; j < index.numInstances(); j++) {
                    if (index.instance(j).value(i) > 0)
                        containingTerm++;
                }
                idf = Math.log(((double) index.numInstances() - (double) containingTerm + 0.5) / ((double) containingTerm + 0.5));
                idfs.put(termName, idf);
            }
            
            // score(D,Q) = sum i=1 to n ( IDF(qi) * ((f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl ))))
            // f(qi,D) is the term q's frequency in the document D
            // |D| is the length of the document in words
            // avgdl is the average document length in the collection
            score += idf * termFrequency * (k1 + 1) / (termFrequency + k1 * (1 - b + b * Ds.get(index.instance(queryIndex).stringValue(0)) * avgdl));
        }
        return score;
    }
    
    private class Rank implements Comparable<Rank> {

        public Double score;
        public String documentID;
        public Rank(Double similarity, String documentID) {
            this.score = similarity;
            this.documentID = documentID;
        }

        @Override
        public int compareTo(Rank anotherRank) {
            if (anotherRank == null)
                return 1;
            return score.compareTo(anotherRank.score);
        }
    }
}
