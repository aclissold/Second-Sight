#!/usr/bin/env python
from __future__ import print_function

from sklearn.feature_extraction.text import CountVectorizer

def main():
    vectorizer = CountVectorizer(input='filename')

    lyrics = ['lyrics/01-soft-offering.txt',
              'lyrics/02-gold-teeth.txt',
              'lyrics/03-dream.txt',
              'lyrics/04-what-arrows.txt',
              'lyrics/05-promise.txt',
              'lyrics/06-kid-gloves.txt',
              'lyrics/07-neon-beyond.txt',
              'lyrics/08-kintsukuroi.txt',
              'lyrics/09-cathedral-bells.txt',
              'lyrics/10-alcatraz.txt',
              'lyrics/11-harriet.txt',
              'lyrics/12-trishs-song.txt']

    X = vectorizer.fit_transform(lyrics)

    print('Document-term matrix:')
    print(X.toarray())

if __name__ == '__main__':
    main()
