(in-package :cl-nlp)

(defun grad (f x)
  "Approximate the gradient of objective function f at x"
  (let* ((eps (sqrt double-float-epsilon))
	 (n (array-total-size x))
	 (gf (make-array (array-dimensions x))))
    (dotimes (i n)
      (let ((x+1 (alexandria:copy-array x))
	    (x-1 (alexandria:copy-array x)))
	(incf (row-major-aref x+1 i) eps)
	(decf (row-major-aref x-1 i) eps)
	(setf (row-major-aref gf i) (/ (- (funcall f x+1) (funcall f x-1)) 2 eps))))
    gf))

(defun jacobian (c x)
  "Approximate the Jacobian of the constraint function c at x"
  (let* ((n (array-total-size x))
	 (c0 (funcall c x))
	 (m (array-total-size c0))
	 (jac (make-array (list m n)))
	 (eps (sqrt double-float-epsilon))
	 (x+1 (alexandria:copy-array x)))
    (dotimes (i n)
      (let ((tmp (row-major-aref x+1 i)))
	(incf (row-major-aref x+1 i) eps)
	(let ((c+1 (funcall c x+1)))
	  (dotimes (j m)
	    (setf (aref jac j i) (/ (- (row-major-aref c+1 j) (row-major-aref c0 j)) eps))))
	(setf (row-major-aref x+1 i) tmp)))
    jac))

(defun hessian (f x)
  "Approximate the Hessian of scalar function f at x"
  (jacobian #'(lambda (x) (grad f x)) x))

