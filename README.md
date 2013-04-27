text-retrieval-probabilistic
============================

TextRetrieval java program for indexing and similarity measurement using probabilistic methods.
The ranking used is an implementation of the Okapi BM25 retrieval model. The values for the parameters
k1 and b were chosen as 1.2 and 0.75 as literature recommends to choose in absence of optimizations.

Building:
The project files retrieved from the repository already contain a build in addition to the source,
so the program can be run immediately. For building the program yourself, install maven (http://maven.apache.org/) 
and 'run mvn clean package'.

Running:
To run the program, got to the TextRetrieval folder and execute 'java -jar target/TextRetrieval-Probabilistic-1.0.0-SNAPSHOT.jar'.
This will bring up the help text for a detailed description of the commands and their options.

The provided script 'run.example.sh' contains examples for each command with some options that can be provided.
