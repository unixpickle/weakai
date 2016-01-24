package main

import (
	"strconv"
	"strings"
)

type Node struct {
	IsLeaf    bool
	LeafClass int

	Question *Question
	Branches map[string]*Node
}

func GenerateIDTree(d DataSet) *Node {
	return generateNode(d, d.Classes(), map[*Question]bool{})
}

func (n *Node) String() string {
	if n.IsLeaf {
		return "Class " + strconv.Itoa(n.LeafClass)
	}

	res := n.Question.Prompt + "\n"
	isFirst := true
	for answer, branch := range n.Branches {
		if !isFirst {
			res += "\n"
		}
		isFirst = false
		branchStr := " " + answer + " -> " + branch.String()
		res += strings.Replace(branchStr, "\n", "\n  ", -1)
	}

	return res
}

func generateNode(d DataSet, allClasses []int, usedQuestions map[*Question]bool) *Node {
	if hom, class := d.homogeneous(); hom {
		return &Node{IsLeaf: true, LeafClass: class}
	}

	question := bestQuestion(d, allClasses, usedQuestions)
	if question == nil {
		return nil
	}

	res := &Node{Question: question, Branches: map[string]*Node{}}
	usedQuestions[question] = true
	for _, answer := range question.Answers {
		subset := d.filter(question, answer)
		subnode := generateNode(subset, allClasses, usedQuestions)
		if subnode == nil {
			return nil
		}
		res.Branches[answer] = subnode
	}
	usedQuestions[question] = false

	return res
}

func bestQuestion(d DataSet, allClasses []int, usedQuestions map[*Question]bool) *Question {
	var leastDisorder float64
	var bestQuestion *Question
	for _, question := range d.Questions() {
		if usedQuestions[question] {
			continue
		}
		var disorder float64
		for _, answer := range question.Answers {
			subset := d.filter(question, answer)
			fractionOfWhole := float64(len(subset)) / float64(len(d))
			disorder += fractionOfWhole * subset.disorder(allClasses)
		}
		if disorder < leastDisorder || bestQuestion == nil {
			leastDisorder = disorder
			bestQuestion = question
		}
	}
	return bestQuestion
}
