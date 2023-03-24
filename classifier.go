package main

import(
    "bufio"
	"fmt"
	"log"
    "math"
    "os"
    "strings"
	"strconv"
    "sync"
)


type Classifier struct {
    realDictionary    map[string]int
    spamDictionary    map[string]int
    totalDictionary   map[string]int
    probabilities     map[string][2]float64
    ProbReal          float64
    ProbSpam          float64
    realTotalWords    int
    spamTotalWords    int
    realTotalMessages int
    spamTotalMessages int
    totalWords        float64
    truePositive      float64
    falsePositive     float64
    trueNegative      float64
    falseNegative     float64
	mutex			  sync.Mutex
}

/* 
	initializeDictionary: This function takes in an input file and initialized the dictionaries
   	of the Classifier struct and depending on if the input file is a ham(real) or spam text
   	populates the respective dictionary, increments the total amount of words found in the file
   	and the number of lines in parallel.
   	ARGS:
		filename: the name of the input file
		isReal: a bool representing if the input file is real or spam text
		wg *sync.WaitGroup: the wait group that the go routine running this function is.
*/
func (c *Classifier) initializeDictionary(filename string, isReal bool, wg *sync.WaitGroup) {
	//ensures that the waitgroup.done is called at function end.
	defer wg.Done()
	//open the input file, update err for error checking.
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	//ensure the file closes at the end of the function
	defer file.Close()
	//create a new scanner to read the file line by line.
	scanner := bufio.NewScanner(file)
	/* 
		Loop through the file line by line, split the line into words
	   	and then store them into a slice. We then loop over the slice
	   	aquiring the mutex lock before we increment the count of the word found
	   	in the dictionary, and the total amount of words in the file and lines.
	   	Finally, we increment the total amount of words found in both files, before we 
	   	release the mutex lock.
	*/
	for scanner.Scan() {
		line := scanner.Text()
		words := strings.Fields(line)
		for _, word := range words {
			c.mutex.Lock()
			c.totalDictionary[word]++
			if isReal {
				c.realDictionary[word]++
				c.realTotalWords++
				c.realTotalMessages++
			} else {
				c.spamDictionary[word]++
				c.spamTotalWords++
				c.spamTotalMessages++
			}//end if
            c.totalWords++
			c.mutex.Unlock()
		}//end for range
	}//end for
	//check for errors with the scanner
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

/*
	calculateProbabilities: This function calculates the probability of each word in the
	real and spam dictionaries and stores the probabilities in the Classifier's probabilities map.
	ARGS:
		smoothing: an int representing the smoothing factor for Laplace smoothing.
*/
func (c *Classifier) calculateProbabilities(smoothing int) {
	// Get the sizes of the real and spam dictionaries
	num_real_words := c.realTotalWords
	num_spam_words := c.spamTotalWords

	// Combine the real and spam dictionaries into one dictionary
	alldict := make(map[string]int)
	for word, count := range c.realDictionary {
		alldict[word] += count
	}
	for word, count := range c.spamDictionary {
		alldict[word] += count
	}

	// Split the alldict map into multiple sub-maps
	numSubMaps := 4 
	subMaps := make([]map[string]int, numSubMaps)
	for i := 0; i < numSubMaps; i++ {
		subMaps[i] = make(map[string]int)
	}
	i := 0
	for word, count := range alldict {
		subMaps[i][word] = count
		i = (i + 1) % numSubMaps
	}
	/* 	
		initializes a WaitGroup and loops through numSubMaps subMaps, adding 1 to the WaitGroup for each. 
	  	It then launches a new goroutine for each subMap, which iterates over the keys in the subMap and calculates the probability for each word
		based on the contents of c.realDictionary and c.spamDictionary. The probabilities are stored in c.probabilities using a mutex lock,
		and the WaitGroup is marked as done when the function completes.
	*/
	var wg sync.WaitGroup
	for i := 0; i < numSubMaps; i++ {
		wg.Add(1)
		go func(subMap map[string]int) {
			//ensure that waitgroup.Done is called at end of function.
			defer wg.Done()
			//this loop checks if the word is in 1 dictionary or both
			for word := range subMap {
				c.mutex.Lock()
				if c.realDictionary[word] == 0 && c.spamDictionary[word] != 0 {
					numerator := float64(smoothing)
					denominator := float64(num_real_words) + (float64(smoothing) * float64(len(alldict)))
					c.probabilities[word] = [2]float64{numerator / denominator, float64(smoothing + c.spamDictionary[word]) / (float64(num_spam_words) + (float64(smoothing) * float64(len(alldict))))}
				} else if c.realDictionary[word] != 0 && c.spamDictionary[word] == 0 {
					numerator := float64(smoothing)
					denominator := float64(num_spam_words) + (float64(smoothing) * float64(len(alldict)))
					c.probabilities[word] = [2]float64{float64(smoothing + c.realDictionary[word]) / (float64(num_real_words) + (float64(smoothing) * float64(len(alldict)))), numerator / denominator}
				} else {
					numerator1 := float64(smoothing + c.realDictionary[word])
					denominator1 := float64(num_real_words) + (float64(smoothing) * float64(len(alldict)))
					numerator2 := float64(smoothing + c.spamDictionary[word])
					denominator2 := float64(num_spam_words) + (float64(smoothing) * float64(len(alldict)))
					c.probabilities[word] = [2]float64{numerator1 / denominator1, numerator2 / denominator2}
				}
				c.mutex.Unlock()
			}
		}(subMaps[i])
	}
	wg.Wait()
}

/*
	calculateProbabilities: This function calculates the probability of each word in the
	real and spam dictionaries and stores the probabilities in the Classifier's probabilities map.
	ARGS:
		smoothing: an int representing the smoothing factor for Laplace smoothing.
*/
func (c *Classifier) classProbCalc(smoothing int) {
	// calculate probability of real
	real_numerator := float64(c.realTotalMessages + smoothing)
	denominator := float64(c.realTotalMessages + c.spamTotalMessages + (smoothing * 2))
	c.ProbReal = real_numerator / denominator
	// calculate probability of spam
	spam_numerator := float64(c.spamTotalMessages + smoothing)
	c.ProbSpam = spam_numerator / denominator
}

/*
	classifyFile: This function reads in a file and classifies it as either real or spam.
	The function uses the Classifier's dictionaries and probabilities map to calculate the probability
	of each word in the file being in either the real or spam dictionary. The function then logs
	the classification and increments the appropriate metric count in the Classifier struct.
	ARGS:
		name: a string representing the name of the file to be classified.
		isReal: a bool representing whether the file is a real message or a spam message.
		wg: a pointer to a WaitGroup object used to synchronize goroutines.
*/
func (c *Classifier) classifyFile(name string, isReal bool, wg *sync.WaitGroup) {
	//ensures that the waitgroup.done is called at function end.
	defer wg.Done()
	//open the input file, update err for error checking.
	file, err := os.Open(name)
	if err != nil {
		log.Fatalf("Error: cannot open file %v\n", err)
	}
	//ensure the file closes at the end of the function
	defer file.Close()

	//create a new scanner to read the file line by line.
	scanner := bufio.NewScanner(file)
	/*
		Loop through each line of a file using a scanner, split the line into words and store them in a slice. 
		Then, for each word in the slice, acquire a mutex lock before checking whether the word exists in either the real or spam dictionary. 
		Based on this, calculate the probability of the word being either real or spam using the Classifier's probabilities map, 
		and add it to the log probabilities for real and spam. Depending on whether the file being classified is real or spam, 
		update the appropriate counters for true positives, false positives, true negatives, and false negatives. Finally, release the mutex lock.
	*/
	for scanner.Scan() {
		line := scanner.Text()
        notIn := math.Log(1.0 / float64(c.totalWords))
		logPReal := math.Log(c.ProbReal)
		logPSpam := math.Log(c.ProbSpam)

		words := strings.Fields(line)

		for _, word := range words {
			if _, ok := c.probabilities[word]; !ok {
				continue
			}
			//check if the word is in either dictionary
			if c.realDictionary[word] == 0 && c.spamDictionary[word] == 0 {
				logPReal += notIn
				logPSpam += notIn
			} else {
				// both dictionaries contain the word
				logPReal += math.Log(c.probabilities[word][0])
				logPSpam += math.Log(c.probabilities[word][1])
			}// end if
		}//end for range
		c.mutex.Lock()
        if isReal {
            if logPReal > logPSpam {
                //fmt.Println("real", logPReal, logPSpam)
                c.truePositive++
            } else {
                //fmt.Println("spam", logPReal, logPSpam)
                c.falseNegative++
            }
        } else {
            if logPReal > logPSpam {
                //fmt.Println("real", logPReal, logPSpam)
                c.falsePositive++
            } else {
                //fmt.Println("spam", logPReal, logPSpam)
                c.trueNegative++
            }
        }//end if
		c.mutex.Unlock()
	}//end for

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error scanning file: %v\n", err)
	}
}

func main() {
	/*
		ARGS:
			Args[1]: real text file used for training.
			Args[2]: spam text file used for training.
			Args[3]: real text file used for validation.
			Args[4]: spam text file used for validation.
			Args[5]: integer for Lalace smoothing.
	*/
	real_training := os.Args[1]
    spam_training := os.Args[2]
	real_valid := os.Args[3]
	spam_valid := os.Args[4]
	smoothing, err := strconv.Atoi(os.Args[5])
	if err != nil {
		log.Fatal(err)
	}
    

    //create classifier
    classifier := Classifier{
		realDictionary:    make(map[string]int),
		spamDictionary:    make(map[string]int),
		totalDictionary:   make(map[string]int),
		probabilities:     make(map[string][2]float64),
		ProbReal:          0.0,
		ProbSpam:          0.0,
		realTotalWords:    0,
		spamTotalWords:    0,
		realTotalMessages: 0,
		spamTotalMessages: 0,
		totalWords:        0.0,
		truePositive:      0.0,
		falsePositive:     0.0,
		trueNegative:      0.0,
		falseNegative:     0.0,
	}

	//initilizes the dictionaries in parallel
	var wg sync.WaitGroup
	wg.Add(2)
	go classifier.initializeDictionary(real_training, true, &wg)
	go classifier.initializeDictionary(spam_training, false, &wg)
	wg.Wait()
	//calculate the probabilities
    classifier.calculateProbabilities(smoothing)
	classifier.classProbCalc(smoothing)
	//classifies the files in parallel
	var wg2 sync.WaitGroup
	wg2.Add(2)
    go classifier.classifyFile(real_valid, true, &wg2)
    go classifier.classifyFile(spam_valid, false, &wg2)
	wg2.Wait()
	//calculates the specificity, sensitivity, and accuracy of the classifier
    totalSize := float64(classifier.truePositive + classifier.trueNegative + classifier.falseNegative + classifier.falsePositive)
    specificity := float64(classifier.trueNegative) / float64(classifier.trueNegative + classifier.falsePositive)
    sensitivity := float64(classifier.truePositive) / float64(classifier.truePositive + classifier.falseNegative)
    accuracy := (float64(classifier.truePositive) + float64(classifier.trueNegative)) / totalSize
    fmt.Printf("specificity: %f, sensitivity: %f, accuracy:  %f\n", specificity, sensitivity, accuracy)
}