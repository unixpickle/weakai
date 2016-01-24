package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: celebrity_lookup <wikipedia_pages.txt>")
		os.Exit(1)
	}

	contents, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	lines := strings.Split(string(contents), "\n")
	fmt.Println("Name,Age,Children,Sings,Acts,Active,HasBeenMarried,Female")
	for _, u := range lines {
		trimmed := strings.TrimSpace(u)
		if len(trimmed) == 0 {
			continue
		}
		info, err := fetchCelebrityInfo(trimmed)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
		} else {
			fmt.Println(info)
		}
	}
}

type CelebrityInfo struct {
	Name           string
	Age            int
	Children       int
	Sings          bool
	Acts           bool
	Active         bool
	HasBeenMarried bool
	Female         bool
}

func (c CelebrityInfo) String() string {
	return fmt.Sprintf("%s,%d,%d,%v,%v,%v,%v,%v", c.Name, c.Age, c.Children,
		c.Sings, c.Acts, c.Active, c.HasBeenMarried, c.Female)
}

func fetchCelebrityInfo(wikipediaURL string) (*CelebrityInfo, error) {
	resp, err := http.Get(wikipediaURL)
	if err != nil {
		return nil, err
	}
	contents, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, err
	}

	source := string(contents)
	res := &CelebrityInfo{}

	nameExp := regexp.MustCompile(`<table class="infobox(.|\n)*?<span.*?>(.*?)</span>`)
	name := nameExp.FindStringSubmatch(source)
	if name == nil {
		return nil, errors.New("could not extract name: " + wikipediaURL)
	}
	res.Name = name[2]

	bornExp := regexp.MustCompile(">Born(.|\n)*?<td>(.|\n)*?\\(age(&#160;|\\s*)([0-9]*)")
	born := bornExp.FindStringSubmatch(source)
	if born == nil {
		return nil, errors.New("could not extract age: " + wikipediaURL)
	}
	res.Age, _ = strconv.Atoi(born[4])

	occupationsExp := regexp.MustCompile(">Occupation(\\(s\\)|s)?((.|\n)*?)</tr>")
	occupations := occupationsExp.FindStringSubmatch(source)
	if occupations != nil {
		if ok, _ := regexp.MatchString("[sS]inger", occupations[2]); ok {
			res.Sings = true
		}
		if ok, _ := regexp.MatchString("[Aa]ct(or|ress)", occupations[2]); ok {
			res.Acts = true
		}
	}

	childrenExp := regexp.MustCompile(">Children(.|\n)*?<td>\\s*([0-9]*)")
	children := childrenExp.FindStringSubmatch(source)
	if children != nil {
		res.Children, _ = strconv.Atoi(children[2])
	}

	res.HasBeenMarried, _ = regexp.MatchString(">Spouse(\\(s\\))?<", source)
	res.Active, _ = regexp.MatchString(">Year.(&#160;|\n)active(.|\n)*?<td>[^<]*present\\s*<",
		source)

	text := strings.ToLower(articleText(source))
	res.Female = femaleScore(text) > maleScore(text)

	return res, nil
}

func articleText(source string) string {
	paragraphExp := regexp.MustCompile("<p>(.*?)</p>")
	matches := paragraphExp.FindAllStringSubmatch(source, -1)
	res := ""
	for _, m := range matches {
		paragraphContents := m[1]
		tagMatcher := regexp.MustCompile("<.*?>")
		res += " " + tagMatcher.ReplaceAllString(paragraphContents, "")
	}
	return res
}

func maleScore(text string) int {
	exp := regexp.MustCompile("\\s(he|him|his)(\\s|\\.|,|;|\\!|\\?)")
	return len(exp.FindAllString(text, -1))

}

func femaleScore(text string) int {
	exp := regexp.MustCompile("\\s(she|her|hers)(\\s|\\.|,|;|\\!|\\?)")
	return len(exp.FindAllString(text, -1))
}
