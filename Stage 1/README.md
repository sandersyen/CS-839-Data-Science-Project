# CS-839-Project-Stage-1

### Tasks:
1. Collect 300 text documents from which we will extract mentions of ONE entity type (e.g., person names, locations, organizations, etc.). These documents contain well-formed sentences and plain text in English (such as those in news articles).
2. We decide on the entity type person names that we will extract from these documents.
3. Go through the documents and mark up the mentions of this entity type.
4. Let this set of documents be B. Split it into a set I of 200 documents and a set J of the remaining 100 documents (after we have randomised the order of the documents). The set I will be used for development (dev set), and the set J will be used for reporting the accuracy of your extractor (the test set).
5. Develop an extractor that achieves at least precision of 90% or higher and as high recall as possible, but at least 60% in recall.
6. The learning step: we start out by building the best possible learning-based extractor:
    * Perform cross validation (CV) on the set I to select the best classifier. We consider at least the following classifiers: decision tree, random forest, support vector machine, linear regression, and logistic regression. We use the scikit-learn package for this CV purpose.
    * Let the best classifier found as above be M. Then debug M using the same set I. 
    * Once we have debugged M, we redo CV to see if another classifier may happen to be more accurate this time. Then we debug that classifier and so on.
    * Finally, Let the resulting classifier be X. We apply X to the set-aside test set J to see if its accuracy already meets the requirement (90% P and 60% R). 
7. **The rule-based postprocessing step**: at this point we  try to add rules in a post-processing step in order to further improve the accuracy of the classifier X.
    * We again use the set I for this purpose. Again, split I into P and Q. Train X on P and apply X to Q. Again, analyze the false pos/neg examples in Q and see what rules we add to fix those.
    * Do this until the overall accuracy of X is already meeting the requirement.
8. Let Y be the final combination of X and the postprocessing rules. Then apply Y to the set-aside test set J and report Y's accuracy on that.