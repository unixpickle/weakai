package main

import (
	"errors"
	"net/http"
	"strings"
	"sync"

	"github.com/yhat/scrape"
	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

const routineCount = 8

type IndexedPage struct {
	URL          string         `json:"url"`
	KeywordCount map[string]int `json:"keywords"`
}

func IndexWikipedia(keywords []string, maxCount int, startPage string) ([]IndexedPage, error) {
	visited := []string{}
	toVisit := []string{startPage}
	res := make([]IndexedPage, 0, maxCount)

	for len(toVisit) > 0 {
		indexes, branches, err := indexPages(keywords, toVisit)
		if err != nil {
			return nil, err
		}
		res = append(res, indexes...)
		visited = append(visited, toVisit...)
		toVisit = make([]string, 0, len(branches))

	AddBranchLoop:
		for _, branch := range branches {
			for _, v := range visited {
				if v == branch {
					continue AddBranchLoop
				}
			}
			toVisit = append(toVisit, branch)
		}

		maxCount := maxCount - len(visited)
		if len(toVisit) > maxCount {
			toVisit = toVisit[:maxCount]
		}
	}

	return res, nil
}

func indexPages(keywords, pages []string) (ind []IndexedPage, branches []string, err error) {
	failChan := make(chan struct{})
	errorChan := make(chan error, 1)

	var resultsLock sync.Mutex
	ind = make([]IndexedPage, 0, len(pages))
	branches = []string{}

	pageChan := make(chan string, len(pages))
	for _, p := range pages {
		pageChan <- p
	}
	close(pageChan)

	var wg sync.WaitGroup
	for i := 0; i < routineCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-failChan:
					return
				default:
				}

				page, ok := <-pageChan
				if !ok {
					return
				}

				res, newBranches, err := indexPage(keywords, page)
				if err != nil {
					select {
					case errorChan <- err:
						close(failChan)
					default:
					}
					return
				}
				resultsLock.Lock()
				ind = append(ind, IndexedPage{page, res})
				branches = addBranches(branches, newBranches)
				resultsLock.Unlock()
			}
		}()
	}

	wg.Wait()
	close(errorChan)
	if err := <-errorChan; err != nil {
		return nil, nil, err
	} else {
		return ind, branches, nil
	}
}

func indexPage(keywords []string, page string) (ind map[string]int, branches []string, err error) {
	resp, err := http.Get(page)
	if err != nil {
		return
	}
	root, err := html.Parse(resp.Body)
	resp.Body.Close()
	if err != nil {
		return
	}

	content, ok := scrape.Find(root, scrape.ById("bodyContent"))
	if !ok {
		return nil, nil, errors.New("no bodyContent element")
	}

	paragraphs := scrape.FindAll(content, scrape.ByTag(atom.P))
	pageText := ""
	for _, p := range paragraphs {
		pageText += elementInnerText(p) + " "
	}
	words := strings.Fields(strings.ToLower(pageText))

	keywordsSet := map[string]bool{}
	for _, kw := range keywords {
		keywordsSet[kw] = true
	}

	ind = map[string]int{}
	for _, word := range words {
		if keywordsSet[word] {
			ind[word] = ind[word] + 1
		}
	}

	links := findWikiLinks(content)
	branches = make([]string, len(links))
	for i, link := range links {
		branches[i] = "https://en.wikipedia.org" + link
	}
	return
}

func elementInnerText(node *html.Node) string {
	text := ""
	node = node.FirstChild
	for node != nil {
		if node.Type == html.TextNode {
			text += node.Data
		} else {
			text += elementInnerText(node)
		}
		node = node.NextSibling
	}
	return text
}

func findWikiLinks(node *html.Node) []string {
	links := scrape.FindAll(node, scrape.ByTag(atom.A))
	res := make([]string, 0, len(links))
	for _, link := range links {
		var u string
		for _, attr := range link.Attr {
			if strings.ToLower(attr.Key) == "href" {
				u = attr.Val
				break
			}
		}
		if strings.HasPrefix(u, "/wiki/") {
			res = append(res, u)
		}
	}
	return res
}

func addBranches(branches, newBranches []string) []string {
AddBranchLoop:
	for _, branch := range newBranches {
		for _, b := range branches {
			if b == branch {
				continue AddBranchLoop
			}
		}
		branches = append(branches, branch)
	}
	return branches
}
