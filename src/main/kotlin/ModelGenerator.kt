package com.ydy.study

import net.razorvine.pickle.Unpickler
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import java.io.File
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.sentenceiterator.SentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory

fun main(args: Array<String>) {
    val models = File("datalemm/").walk().drop(1).map{
        println(it)
        it.readLines()
    }.flatten().toList()

    val iter = CollectionSentenceIterator(models)
    val t = DefaultTokenizerFactory()
    t.tokenPreProcessor = CommonPreprocessor()
    val vec = Word2Vec.Builder()
        .minWordFrequency(5)
        .layerSize(100)
        .batchSize(10000)
        .seed(42)
        .windowSize(5)
        .iterate(iter)
        .epochs(5)
        .tokenizerFactory(t)
        .build()
    vec.fit()



// TODO:나중에 TFIDF도 해보자
//    val vectorizer = TfidfVectorizer.Builder()
//        .setMinWordFrequency(3)
//        .setStopWords(listOf<String>())
//        .setTokenizerFactory(t)
//        .setIterator(iter).build()
    print("first (quit to fin): ")
    var first = readLine()
    while(first != "quit") {
        print("second: ")
        var second = readLine()
        println("similarity of $first and $second: ${vec.similarity(first, second)}")
        println("similar to $first: ${vec.wordsNearest(first, 10)}")
        print("first: ")
        first = readLine()
    }
}