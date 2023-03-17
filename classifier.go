package main

import(
	"fmt"
    "bufio"
    "math"
    "log"
    "os"
    "strings"
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
}

func (c *Classifier) initializeDictionary(filename string, isReal bool, wg *sync.WaitGroup) {
	defer wg.Done()

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		words := strings.Fields(line)
		for _, word := range words {
			c.totalDictionary[word]++
			if isReal {
				c.realDictionary[word]++
				c.realTotalWords++
				c.realTotalMessages++
			} else {
				c.spamDictionary[word]++
				c.spamTotalWords++
				c.spamTotalMessages++
			}
            c.totalWords++
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func (c *Classifier) calculateProbabilities(smoothing int) {
	// Get the sizes of the real and spam dictionaries
	words1 := c.realTotalWords
	words2 := c.spamTotalWords

	// Combine the real and spam dictionaries into one dictionary
	alldict := make(map[string]int)
	for word, count := range c.realDictionary {
		alldict[word] += count
	}
	for word, count := range c.spamDictionary {
		alldict[word] += count
	}

	// Split the alldict map into multiple sub-maps
	numSubMaps := 4 // You can adjust this to optimize for your specific hardware
	subMaps := make([]map[string]int, numSubMaps)
	for i := 0; i < numSubMaps; i++ {
		subMaps[i] = make(map[string]int)
	}
	i := 0
	for word, count := range alldict {
		subMaps[i][word] = count
		i = (i + 1) % numSubMaps
	}

	// Calculate the probabilities for each word in parallel
	var wg sync.WaitGroup
	for i := 0; i < numSubMaps; i++ {
		wg.Add(1)
		go func(subMap map[string]int) {
			defer wg.Done()
			for word, count := range subMap {
				if c.realDictionary[word] == 0 && c.spamDictionary[word] != 0 {
					// ham doesn't have it
					numerator := float64(smoothing)
					denominator := float64(words1) + (float64(smoothing) * float64(len(alldict)))
					c.probabilities[word] = [2]float64{numerator / denominator, float64(smoothing + c.spamDictionary[word]) / (float64(words2) + (float64(smoothing) * float64(len(alldict))))}
				} else if c.realDictionary[word] != 0 && c.spamDictionary[word] == 0 {
					// spam doesn't have it
					numerator := float64(smoothing)
					denominator := float64(words2) + (float64(smoothing) * float64(len(alldict)))
					c.probabilities[word] = [2]float64{float64(smoothing + c.realDictionary[word]) / (float64(words1) + (float64(smoothing) * float64(len(alldict)))), numerator / denominator}
				} else {
					// both have it
					numerator1 := float64(smoothing + c.realDictionary[word])
					denominator1 := float64(words1) + (float64(smoothing) * float64(len(alldict)))
					numerator2 := float64(smoothing + c.spamDictionary[word])
					denominator2 := float64(words2) + (float64(smoothing) * float64(len(alldict)))
					c.probabilities[word] = [2]float64{numerator1 / denominator1, numerator2 / denominator2}
				}
			}
		}(subMaps[i])
	}
	wg.Wait()
}

func (c *Classifier) classProbCalc(smoothing int) {
	numerator1 := float64(c.realTotalMessages + smoothing)
	denominator := float64(c.realTotalMessages + c.spamTotalMessages + (smoothing * 2))
	c.ProbReal = numerator1 / denominator
	numerator2 := float64(c.spamTotalMessages + smoothing)
	c.ProbSpam = numerator2 / denominator
}

func (c *Classifier) classifyFile(name string, isReal bool) {
	file, err := os.Open(name)
	if err != nil {
		log.Fatalf("Error: cannot open file %v\n", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

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

			if c.realDictionary[word] == 0 && c.spamDictionary[word] == 0 {
				// neither have it.
				logPReal += notIn
				logPSpam += notIn
			} else {
				// both have it
				logPReal += math.Log(c.probabilities[word].first)
				logPSpam += math.Log(c.probabilities[word].second)
			}
		}
        if isReal {
            if logPReal > logPSpam {
                fmt.Println("real", logPReal, logPSpam)
                c.truePositive++
            } else {
                fmt.Println("spam", logPReal, logPSpam)
                c.falseNegative++
            }
        } else {
            if logPReal > logPSpam {
                fmt.Println("real", logPReal, logPSpam)
                c.falsePositive++
            } else {
                fmt.Println("spam", logPReal, logPSpam)
                c.trueNegative++
            }
        }
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("Error scanning file: %v\n", err)
	}
}

func main() {
	real_training := os.Args[1]
    spam_training := os.Args[2]
	real_valid := os.Args[3]
	spam_valid := os.Args[4]
	smoothing := os.Args[5]
    
    user_test := os.Args[6]


    // create classifier
    classifier := Classifier{
		realDictionary:  make(map[string]int),
		spamDictionary:  make(map[string]int),
		totalDictionary: make(map[string]int),
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go classifier.initializeDictionary(real_training, true, &wg)
	go classifier.initializeDictionary(spam_training, false, &wg)
	wg.Wait()

    classifier.calculateProbabilities(smoothing)
    fmt.Println("----------Validation----------")
    classifier.classifyFile(real_valid, true)
    classifier.classifyFile(spam_valid, false)
    fmt.Println("------------------------------")
    classifier.classifyFile(user_test, true)
    totalSize := float64(classifier.truePositive + classifier.trueNegative + classifier.falseNegative + classifier.falsePositive)
    specificity := float64(classifier.trueNegative) / float64(classifier.trueNegative + classifier.falsePositive)
    sensitivity := float64(classifier.truePositive) / float64(classifier.truePositive + classifier.falseNegative)
    accuracy := (float64(classifier.truePositive) + float64(classifier.trueNegative)) / totalSize
    fmt.Printf("%f %f %f\n", specificity, sensitivity, accuracy)

}