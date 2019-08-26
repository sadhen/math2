/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */


package com.tencent.angel.ml.math2.matrix;

import com.tencent.angel.ml.math2.utils.MathException;
import com.tencent.angel.ml.math2.storage.IntDoubleDenseVectorStorage;
import com.tencent.angel.ml.math2.ufuncs.expression.Add;
import com.tencent.angel.ml.math2.ufuncs.expression.Axpy;
import com.tencent.angel.ml.math2.ufuncs.expression.Binary;
import com.tencent.angel.ml.math2.ufuncs.expression.Div;
import com.tencent.angel.ml.math2.ufuncs.expression.Mul;
import com.tencent.angel.ml.math2.ufuncs.expression.SAdd;
import com.tencent.angel.ml.math2.ufuncs.expression.SDiv;
import com.tencent.angel.ml.math2.ufuncs.expression.SMul;
import com.tencent.angel.ml.math2.ufuncs.expression.SSub;
import com.tencent.angel.ml.math2.ufuncs.expression.Sub;
import com.tencent.angel.ml.math2.ufuncs.expression.Unary;
import com.tencent.angel.ml.math2.utils.VectorUtils;
import com.tencent.angel.ml.math2.vector.IntDoubleVector;
import com.tencent.angel.ml.math2.vector.Vector;


public abstract class RowBasedMatrix<Vec extends Vector> extends Matrix {

  protected Vec[] rows;
  protected long cols;

  public RowBasedMatrix() {
  }

  public RowBasedMatrix(int matrixId, int clock, long cols, Vec[] rows) {
    this.matrixId = matrixId;
    this.clock = clock;
    this.rows = rows;
    this.cols = cols;
  }

  public abstract RowBasedMatrix calculate(int rowId, Vector other, Binary op);

  public abstract RowBasedMatrix calculate(Vector other, Binary op);

  public abstract RowBasedMatrix calculate(Matrix other, Binary op);

  public abstract RowBasedMatrix calculate(Unary op);

  @Override
  public Vec getRow(int idx) {
    return rows[idx];
  }

  @Override
  public Vec getCol(int idx) {
    throw new MathException("RBMatrix is not support to getCol");
  }

  @Override
  public int getNumRows() {
    return rows.length;
  }

  public abstract void setRow(int idx, Vec v);

  public Vec[] getRows() {
    return rows;
  }

  public abstract void setRows(Vec[] rows);

  public long getDim() {
    return cols;
  }

  @Override
  public double sum() {
    double res = 0.0;
    for (int i = 0; i < rows.length; i++) {
      res += rows[i].sum();
    }
    return res;
  }

  @Override
  public double std() {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < rows.length; i++) {
      sum1 += rows[i].sum();
      double tmp = rows[i].norm();
      sum2 += tmp * tmp;
    }

    sum1 /= rows.length * cols;
    sum2 /= rows.length * cols;
    return Math.sqrt(sum2 - sum1 * sum1);
  }

  @Override
  public double average() {
    return sum() / (rows.length * cols);
  }

  @Override
  public double norm() {
    double res = 0.0;
    for (int i = 0; i < rows.length; i++) {
      double tmp = rows[i].norm();
      res += tmp * tmp;
    }
    return Math.sqrt(res);
  }

  @Override
  public Vector dot(Vector other) {
    double[] resArr = new double[rows.length];
    for (int i = 0; i < rows.length; i++) {
      resArr[i] = rows[i].dot(other);
    }
    IntDoubleDenseVectorStorage storage = new IntDoubleDenseVectorStorage(resArr);
    return new IntDoubleVector(matrixId, 0, clock, rows.length, storage);
  }

  @Override
  public Vector transDot(Vector other) {
    Vector res = null;
    for (int i = 0; i < rows.length; i++) {
      if (i == 0) {
        res = rows[i].mul(VectorUtils.getDouble(other, i));
      } else {
        res.iaxpy(rows[i], VectorUtils.getDouble(other, i));
      }
    }
    return res;
  }

  @Override
  public Matrix iadd(int rowId, Vector other) {
    return calculate(rowId, other, new Add(true));
  }

  @Override
  public Matrix add(int rowId, Vector other) {
    return calculate(rowId, other, new Add(false));
  }

  @Override
  public Matrix isub(int rowId, Vector other) {
    return calculate(rowId, other, new Sub(true));
  }

  @Override
  public Matrix sub(int rowId, Vector other) {
    return calculate(rowId, other, new Sub(false));
  }

  @Override
  public Matrix imul(int rowId, Vector other) {
    return calculate(rowId, other, new Mul(true));
  }

  @Override
  public Matrix mul(int rowId, Vector other) {
    return calculate(rowId, other, new Mul(false));
  }

  @Override
  public Matrix idiv(int rowId, Vector other) {
    return calculate(rowId, other, new Div(true));
  }

  @Override
  public Matrix div(int rowId, Vector other) {
    return calculate(rowId, other, new Div(false));
  }

  @Override
  public Matrix iaxpy(int rowId, Vector other, double aplha) {
    return calculate(rowId, other, new Axpy(true, aplha));
  }

  @Override
  public Matrix axpy(int rowId, Vector other, double aplha) {
    return calculate(rowId, other, new Axpy(false, aplha));
  }

  @Override
  public Matrix iadd(Vector other) {
    return calculate(other, new Add(true));
  }

  @Override
  public Matrix add(Vector other) {
    return calculate(other, new Add(false));
  }

  @Override
  public Matrix isub(Vector other) {
    return calculate(other, new Sub(true));
  }

  @Override
  public Matrix sub(Vector other) {
    return calculate(other, new Sub(false));
  }

  @Override
  public Matrix imul(Vector other) {
    return calculate(other, new Mul(true));
  }

  @Override
  public Matrix mul(Vector other) {
    return calculate(other, new Mul(false));
  }

  @Override
  public Matrix idiv(Vector other) {
    return calculate(other, new Div(true));
  }

  @Override
  public Matrix div(Vector other) {
    return calculate(other, new Div(false));
  }

  @Override
  public Matrix iaxpy(Vector other, double aplha) {
    return calculate(other, new Axpy(true, aplha));
  }

  @Override
  public Matrix axpy(Vector other, double aplha) {
    return calculate(other, new Axpy(false, aplha));
  }

  @Override
  public Matrix iadd(Matrix other) {
    return calculate(other, new Add(true));
  }

  @Override
  public Matrix add(Matrix other) {
    return calculate(other, new Add(false));
  }

  @Override
  public Matrix isub(Matrix other) {
    return calculate(other, new Sub(true));
  }

  @Override
  public Matrix sub(Matrix other) {
    return calculate(other, new Sub(false));
  }

  @Override
  public Matrix imul(Matrix other) {
    return calculate(other, new Mul(true));
  }

  @Override
  public Matrix mul(Matrix other) {
    return calculate(other, new Mul(false));
  }

  @Override
  public Matrix idiv(Matrix other) {
    return calculate(other, new Div(true));
  }

  @Override
  public Matrix div(Matrix other) {
    return calculate(other, new Div(false));
  }

  @Override
  public Matrix iaxpy(Matrix other, double aplha) {
    return calculate(other, new Axpy(true, aplha));
  }

  @Override
  public Matrix axpy(Matrix other, double aplha) {
    return calculate(other, new Axpy(false, aplha));
  }

  @Override
  public Matrix iadd(double x) {
    return calculate(new SAdd(true, x));
  }

  @Override
  public Matrix add(double x) {
    return calculate(new SAdd(false, x));
  }

  @Override
  public Matrix isub(double x) {
    return calculate(new SSub(true, x));
  }

  @Override
  public Matrix sub(double x) {
    return calculate(new SSub(false, x));
  }

  @Override
  public Matrix imul(double x) {
    return calculate(new SMul(true, x));
  }

  @Override
  public Matrix mul(double x) {
    return calculate(new SMul(false, x));
  }

  @Override
  public Matrix idiv(double x) {
    return calculate(new SDiv(true, x));
  }

  @Override
  public Matrix div(double x) {
    return calculate(new SDiv(false, x));
  }

  @Override
  public void clear() {
    matrixId = 0;
    clock = 0;
    cols = 0;
    for (int i = 0; i < rows.length; i++) {
      rows = null;
    }
  }

  public void clearRow(int rowId) {
    assert rowId >= 0 && rowId < rows.length;
    rows[rowId] = null;
  }

  public abstract void initEmpty(int idx);
}