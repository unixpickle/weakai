package main

import "math"

type Question struct {
	Prompt  string
	Answers []string
}

type DataEntry interface {
	QuestionAnswers() map[*Question]string
	Class() int
}

type DataSet []DataEntry

func (d DataSet) Questions() []*Question {
	if len(d) == 0 {
		return []*Question{}
	}
	res := make([]*Question, 0, len(d[0].QuestionAnswers()))
	for q := range d[0].QuestionAnswers() {
		res = append(res, q)
	}
	return res
}

func (d DataSet) Classes() []int {
	res := []int{}
	seen := map[int]bool{}
	for _, e := range d {
		if !seen[e.Class()] {
			seen[e.Class()] = true
			res = append(res, e.Class())
		}
	}
	return res
}

func (d DataSet) filter(q *Question, answer string) DataSet {
	res := DataSet{}
	for _, x := range d {
		if x.QuestionAnswers()[q] == answer {
			res = append(res, x)
		}
	}
	return res
}

func (d DataSet) disorder(allClasses []int) float64 {
	classCount := map[int]int{}
	for _, x := range d {
		classCount[x.Class()]++
	}

	// Thank you, information theorists.
	var res float64
	for _, class := range allClasses {
		fraction := float64(classCount[class]) / float64(len(d))
		if fraction != 0 {
			res -= fraction * math.Log(fraction)
		}
	}
	return res
}

func (d DataSet) homogeneous() (homogeneous bool, class int) {
	if len(d) == 0 {
		return true, -1
	}
	firstClass := d[0].Class()
	for i := 1; i < len(d); i++ {
		if d[i].Class() != firstClass {
			return false, 0
		}
	}
	return true, firstClass
}
