(define (problem checker-problem)
  (:domain checker-move)

  (:objects
    c1 c2 c3 c4 c5 - checker
    p1 p2 p3 p4 p5 p6 - position
  )

  (:init
    (checker-at c1 p1)
    (checker-at c2 p3)
    (checker-at c3 p4)
    (checker-at c4 p5)
    (checker-at c5 p6)
    (adjacent p1 p2)
    (adjacent p2 p3)
    (adjacent p3 p4)
    (adjacent p4 p5)
    (adjacent p5 p6)
    (adjacent p2 p1)
    (adjacent p3 p2)
    (adjacent p4 p3)
    (adjacent p5 p4)
    (adjacent p6 p5)

    (= (has-color c1) 0)
    (= (has-color c2) 1)
    (= (has-color c3) 1)
    (= (has-color c4) 1)
    (= (has-color c5) 1)

    (empty p2)
    (less p1 p2)
    (less p2 p3)
    (less p3 p4)
    (less p4 p5)
    (less p5 p6)
  )

  (:goal
    (and
      (checker-at c2 p1)
      (checker-at c3 p2)
      (checker-at c4 p3)
      (checker-at c5 p4)
      (checker-at c1 p6)
    )
  )
)