package com.ydy.study

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.openkoreantext.processor.OpenKoreanTextProcessorJava
//import org.openkoreantext.processor.phrase_extractor.KoreanPhraseExtractor
//import org.openkoreantext.processor.tokenizer.KoreanTokenizer
import net.razorvine.pickle.Unpickler
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.openkoreantext.processor.tokenizer.KoreanTokenizer
import scala.collection.Seq
import java.io.File
// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/SentimentExampleIterator.java
data class Token(val text: String) {
    constructor(text:String, pos:String): this("('$text','$pos')")
    constructor(vararg tokens: Token): this(tokens.joinToString(" "))
    override fun toString() = text
}

class Word2Vec1 {

    companion object {
        val model = WordVectorSerializer.readWord2VecModel("pos-news.kvb")
        val vectorDim = model.vocab.wordAtIndex(0).length
        val wordFreq = Unpickler().load(File("naver_movie_pos_freq.pkl").inputStream())
                as Map<String, Int>
        val FREQ_CONST = 50000

        /**
         * returns weighted sum of tokens in a string
         * tokenization by open korean tokenizer (twitter)
         * weight by sort-of IDF
         * OOV words are silently ignored
         */
        fun sentence2vec(sentence: String, debug:Boolean = false): INDArray? {
            val cleanText = Regex("[^ㄱ-힣a-zA-Z0-9]+").replace(sentence, " ").trim()
            val tokens = mutableListOf<String>()
            val weights = mutableListOf<Float>()
            for(token in OpenKoreanTextProcessorJava.tokenize(cleanText)) {
                val pos = token.pos().toString()
                if (pos!="Josa" && pos!="Eomi") {
                    val word = "('${token.text()}','$pos')"
                    if (model.vocab().containsWord(word)) {
                        tokens.add(word)
                        weights.add(idf(word))
                        if (debug)
                            println("$word ${idf(word)}")
                    }
                }
            }
            if (tokens.isNotEmpty()) {
                val v = model.getWordVectors(tokens)
                val wv = Nd4j.create(weights)
                return wv.mmul(v)
            } else {
                return null
            }
        }

        fun tokens2vec(tokens: Iterable<Token>): INDArray {
            val v = model.getWordVectors( tokens.map {it.toString()})
            val wv = Nd4j.create(tokens.map { idf(it.toString())})
            return normalizei(wv.mmul(v))
        }

        fun idf(word: String): Float {
            return FREQ_CONST.toFloat() / (FREQ_CONST + wordFreq.getOrDefault(word, FREQ_CONST)).toFloat()
        }

        fun tokenize(sentence: String, includeOOV: Boolean = true): MutableList<Token> {
            val cleanText = Regex("[^ㄱ-힣a-zA-Z0-9]+").replace(sentence, " ").trim()
            val result = arrayListOf<Token>()
            for (token in OpenKoreanTextProcessorJava.tokenize(cleanText)) {
                val t = Token(token.text(), token.pos().toString())
                if (includeOOV || model.vocab.containsWord(t.toString()))
                    result.add(t)
            }
            return result
        }

        fun getVectors(tokens: List<Token>): INDArray? {
            return if (tokens.size > 0)
                model.getWordVectors(tokens.map { it.toString() })
            else
                null
        }

        fun normalizei(t: INDArray): INDArray {
            val sqrsum = t.norm2(t.shape().size-1)
            t.divi(sqrsum)
            return t
        }

        fun cosineSim(vec1: INDArray, vec2: INDArray): Double {
            return Transforms.cosineSim(vec1, vec2)
        }
    }
}

//fun main(args: Array<String>) {
//    println("Reading word2vec model...")
//    val w2vModel = WordVectorSerializer.readWord2VecModel("pos-news.kvb")
//    println(w2vModel.wordsNearest("('영화','Noun')", 10))
//    val vector = w2vModel.getWordVector("('영화','Noun')")
//
//    println(Word2Vec.sentence2vec("뭐 볼만한 영화 없어?"))
//    val pos_freq = Unpickler().load(File("naver_movie_pos_freq.pkl").inputStream())
//            as Map<String, Int>
//    println("freq of 영화 = " + pos_freq["('영화','Noun')"])
//}