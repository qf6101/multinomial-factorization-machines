package io.github.qf6101.mfm.factorization.multinomial

import io.github.qf6101.mfm.util.MfmTestSparkSession
import org.scalatest.FunSuite

/**
  * Created by qfeng on 16-9-18.
  */
class MfmCoefficientsSuite extends FunSuite with MfmTestSparkSession {
  test("test MfmCoefficients' += operation") {
    val left = new MfmCoefficients(0.0, 0.01, 1, 1, 1, false, true, false, 2)
    val right = new MfmCoefficients(0.0, 0.01, 1, 1, 1, false, true, false, 2)

    val leftSample = left.thetas(0).w(0)
    val rightSample = right.thetas(0).w(0)
    println(leftSample)
    println(rightSample)
    left += right
    println(left.thetas(0).w(0))
    assert(left.thetas(0).w(0) == leftSample + rightSample)
  }

  test("test MfmCoefficients' + operation") {
    val left = new MfmCoefficients(0.0, 0.01, 1, 1, 1, false, true, false, 2)
    val right = new MfmCoefficients(0.0, 0.01, 1, 1, 1, false, true, false, 2)

    val leftSample = left.thetas(0).w(0)
    val rightSample = right.thetas(0).w(0)
    println(leftSample)
    println(rightSample)
    val sum = left + right
    println(sum.asInstanceOf[MfmCoefficients].thetas(0).w(0))
    assert(sum.asInstanceOf[MfmCoefficients].thetas(0).w(0) == leftSample + rightSample)
  }
}
