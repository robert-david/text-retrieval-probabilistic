java -jar target/TextRetrieval-Probabilistic-1.0.0-SNAPSHOT.jar index --source 20_newsgroups_subset --target 20_newsgroups_subset_RESULT --stemming true -b 500 --lower 3 --upper 19 
java -jar target/TextRetrieval-Probabilistic-1.0.0-SNAPSHOT.jar match --indexFile 20_newsgroups_subset_RESULT/index.arff.gz --source 20_newsgroups_subset --target 20_newsgroups_subset_RESULT
