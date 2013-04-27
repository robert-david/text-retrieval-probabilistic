package at.ac.tuwien.ifs.ir;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.WordTokenizer;

public class Indexer {

    private static Logger log = LoggerFactory.getLogger(Indexer.class);
    
    private static String blockDir = "blocks";
    private static String blockFile = "block";
    private static String extension = ".arff.gz";

    private String source = ".";
    private String target = ".";
    private int blockSize = 100;
    private boolean stemming = false;
    private int upper = -1;
    private int lower = -1;

    public Indexer() {}
    
    public Indexer(String source, String target, int blockSize) {
        this.source = source;
        this.target = target;
        this.blockSize = blockSize;        
    }
    
    public Indexer(String source, String target, int blockSize, boolean stemming, int upper, int lower) {
        this(source, target, blockSize);
        this.stemming = stemming;
        this.upper = upper;
        this.lower = lower;
    }
    
    public void index() {
        log.info("Started indexing ...");
        createBlocks();
        merge();
        log.info("Done indexing");
    }
    
    private void createBlocks() {
        log.info("Started creating blocks ...");
        
        Instances instances = null;
        try {
            log.info("Loading files ...");
            TextDirectoryLoader tdl = new TextDirectoryLoader();
            tdl.setDirectory(new File(source));
            tdl.setRetrieval(Loader.INCREMENTAL);
            tdl.setOutputFilename(true);
            instances = tdl.getDataSet();
        } catch (IOException ioe) {
            log.error("Error loading files", ioe);
            return;
        }

        log.info("Processing contents ...");
        // we implement the dictionary as a hash map, alternative would be tree map
        Map<String, Double[]> dictionary = new HashMap<String,Double[]>();
        WordTokenizer wordTokenizer = new WordTokenizer();
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        wordTokenizer.setDelimiters(wordTokenizer.getDelimiters() + "");
        SnowballStemmer stemmer = new SnowballStemmer();
        
        int entryCounter = 0;
        int blockCounter = 0;
        
        // while not all file content (text attribute) is processed
        while (entryCounter < instances.numInstances()) {

            String fileContent = instances.instance(entryCounter).stringValue(0);
            wordTokenizer.tokenize(fileContent);
            String token;
            
            // iterate through the tokens of the current file content
            while (wordTokenizer.hasMoreElements()) {
                token = wordTokenizer.nextElement().toString().trim().toLowerCase();
                
                // optional stemming
                if (stemming)
                    token = stemmer.stem(token);
                
                // check if dictionary contains term
                if (dictionary.containsKey(token)) {
                    Double[] postingsList = dictionary.get(token);
                    if (postingsList[blockCounter] == null)
                        postingsList[blockCounter] = new Double(1);
                    else
                        postingsList[blockCounter] += 1;
                } else {
                    Double[] postingsList = new Double[blockSize];
                    postingsList[blockCounter] = new Double(1);
                    dictionary.put(token, postingsList);
                }                   
            }
            
            entryCounter++;
            blockCounter++;
            
            // if block size is reached, write the block to disk
            if (entryCounter == instances.numInstances() || blockCounter == blockSize) {

                // optional frequency thresholding
                if (upper > -1 || lower > -1) {
                    List<String> removeList = new ArrayList<String>();
                    for (Iterator<String> i = dictionary.keySet().iterator(); i.hasNext(); ) {
                        String key = i.next();
                        Double[] postingsList = dictionary.get(key);
                        boolean remove = true;
                        for (int j = 0; j < postingsList.length; j++) {                             
                            if (postingsList[j] != null)
                                if((postingsList[j] < lower) ||
                                            (upper > -1 && postingsList[j] > upper))
                                    postingsList[j] = null;
                                else
                                    remove = false;
                        }
                        // all values for the key were thresholded away, so remove the term from the dictionary
                        if (remove)
                            removeList.add(key);
                    }
                    for (String key : removeList)
                        dictionary.remove(key);
                }
                
                log.info("Creating block ...");
                // create a vector for attributes filename, class, and 1 attribute for each term
                FastVector attributes = new FastVector(dictionary.keySet().size() + 2);
                
                attributes.addElement(new Attribute("@nameOfTheDocument", (FastVector) null));
                attributes.addElement(new Attribute("@classOfTheDocument", (FastVector) null));
                
                List<String> sortedTerms = new ArrayList<String>(dictionary.keySet());
                Collections.sort(sortedTerms);

                for (String term : sortedTerms)
                    attributes.addElement(new Attribute(term));
                
                Instances block = null;
                try {
                    block = new Instances(blockFile + entryCounter + extension, attributes, blockSize);
                } catch (IllegalArgumentException iae) {
                    log.error("Error processing terms as attributes", iae);
                    return;
                }
                
                for (int i = 0; i < blockCounter; i++) {
                    int index = entryCounter - blockCounter + i;

                    Instance instance = new Instance(dictionary.keySet().size() + 2);
                    instance.setDataset(block);
                    instance.setValue(0, instances.instance(index).stringValue(1));
                    instance.setValue(1, instances.instance(index).stringValue(2));

                    for (int j = 0; j < sortedTerms.size(); j++) {
                        Double termCount = dictionary.get(sortedTerms.get(j))[i];
                        instance.setValue(2 + j, termCount == null ? 0 : termCount.doubleValue());
                    }
                    block.add(instance);
                }
                
                try {
                    ArffSaver saver = new ArffSaver();
                    saver.setInstances(block);
                    String filename = target + "/" + blockDir + "/" + blockFile + entryCounter + extension;
                    saver.setFile(new File(filename));
                    saver.setCompressOutput(true);
                    saver.writeBatch();
                    log.info("Wrote block to " + filename);
                } catch (IOException ioe) {
                    log.error("Error saving block to ARFF file", ioe);
                    return;
                }

                dictionary.clear();
                blockCounter = 0;
            }
        }
        log.info("Done creating blocks");
    }
    
    private void merge() {
        log.info("Started merging blocks ...");
        
        Instances index = null;
        try {
            FastVector attributes = new FastVector(2);
            attributes.addElement(new Attribute("@nameOfTheDocument", (FastVector) null));
            attributes.addElement(new Attribute("@classOfTheDocument", (FastVector) null));
            index = new Instances("index", attributes, 0);

            File file = new File(target + "/" + blockDir);
            for (String filename : file.list(new FilenameFilter() {

                @Override
                public boolean accept(File file, String filename) {
                    return (filename.startsWith(blockFile) && filename.endsWith(extension)) ? true : false;
                }
            })) {
                log.info("Adding " + filename + " to index ...");

                Instances block = new DataSource(target + "/" + blockDir + "/" + filename).getDataSet();

                // insert new attributes into the index
                @SuppressWarnings("unchecked")
                Enumeration<Attribute> blockAttrEnum = block.enumerateAttributes();
                while (blockAttrEnum.hasMoreElements()) {
                    Attribute blockAttribute = blockAttrEnum.nextElement();
                    Attribute indexAttribute = index.attribute(blockAttribute.name());
                    if (indexAttribute == null) {
                        indexAttribute = new Attribute(blockAttribute.name());
                        @SuppressWarnings("unchecked")
                        Enumeration<Attribute> indexAttrEnum = index.enumerateAttributes();
                        int position = 0;
                        while (indexAttrEnum.hasMoreElements()) {
                            String name = indexAttrEnum.nextElement().name();
                                if ((name.compareTo(blockAttribute.name()) < 0)
                                    || name.equals("@nameOfTheDocument")
                                    || name.equals("@classOfTheDocument"))
                            position++;
                        }
                        index.insertAttributeAt(indexAttribute, position);
                    }
                }

                @SuppressWarnings("unchecked")
                Enumeration<Instance> blockInstEnum = block.enumerateInstances();
                
                while (blockInstEnum.hasMoreElements()) {
                    Instance blockInstance = blockInstEnum.nextElement();

                    // sparse instances do not work correctly
                    // the first attribute value is lost when saving if it is a string
                    // SparseInstance newInstance = new SparseInstance(index.numAttributes());
                    Instance newInstance = new Instance(index.numAttributes());
                    newInstance.setDataset(index);

                    for (int i = 0; i < blockInstance.numAttributes(); i++) {
                        if (blockInstance.attribute(i).name().equals("@nameOfTheDocument")
                                ||blockInstance.attribute(i).name().equals("@classOfTheDocument"))
                            newInstance.setValue(index.attribute(blockInstance.attribute(i).name()), 
                                    blockInstance.stringValue(blockInstance.attribute(i)));
                        else
                            newInstance.setValue(index.attribute(blockInstance.attribute(i).name()), 
                                    blockInstance.value(blockInstance.attribute(i)));
                    }
                    // seems not to work correctly
                    // the missing values of the first instance do not get set 
                    newInstance.replaceMissingValues(new double[newInstance.numAttributes()]);
                    index.add(newInstance);                    
                }
            }
        } catch (Exception e) {
            log.error("Error merging blocks", e);
            return;
        }
        
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(index);
            String filename = target + "/" + "index" + extension;
            saver.setFile(new File(filename));
            saver.setCompressOutput(true);
            saver.writeBatch();
            log.info("Wrote index to " + filename);
        } catch (IOException ioe) {
            log.error("Error saving index to ARFF file", ioe);
            return;
        }
        log.info("Done merging");
    }
}
