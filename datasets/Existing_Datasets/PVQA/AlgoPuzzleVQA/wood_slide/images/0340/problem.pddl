(define (problem wood-slide-puzzle)
  (:domain woodslide)

  (:objects
    b1 b2 b3 b4 b5 b6 b7 b8 b9 - block
    e1 e2 - empty-space
  )

  (:init
    (= (total-moves) 0)
    (= (min_x) 1) (= (min_y) 1)
    (= (max_x) 5) (= (max_y) 4)

    (= (xmin b1) 3) (= (ymin b1) 3)
    (= (xmax b1) 4) (= (ymax b1) 4)

    (= (xmin b2) 2) (= (ymin b2) 2)
    (= (xmax b2) 2) (= (ymax b2) 3)

    (= (xmin b3) 1) (= (ymin b3) 2)
    (= (xmax b3) 1) (= (ymax b3) 3)

    (= (xmin b4) 2) (= (ymin b4) 4)
    (= (xmax b4) 2) (= (ymax b4) 4)

    (= (xmin b5) 1) (= (ymin b5) 4)
    (= (xmax b5) 1) (= (ymax b5) 4)

    (= (xmin b6) 1) (= (ymin b6) 1)
    (= (xmax b6) 2) (= (ymax b6) 1)

    (= (xmin b7) 3) (= (ymin b7) 1)
    (= (xmax b7) 4) (= (ymax b7) 1)

    (= (xmin b8) 5) (= (ymin b8) 1)
    (= (xmax b8) 5) (= (ymax b8) 2)

    (= (xmin b9) 5) (= (ymin b9) 3)
    (= (xmax b9) 5) (= (ymax b9) 4)

    (= (x e1) 3) (= (y e1) 2)
    (= (x e2) 4) (= (y e2) 2)
    (= (dim_type b1) 4)
    (= (dim_type b2) 2)
    (= (dim_type b3) 2)
    (= (dim_type b4) 1)
    (= (dim_type b5) 1)
    (= (dim_type b6) 3)
    (= (dim_type b7) 3)
    (= (dim_type b8) 2)
    (= (dim_type b9) 2)

  )

  (:goal
    (and
      (= (xmin b1) 3) (= (ymin b1) 2)
      (= (xmax b1) 4) (= (ymax b1) 3)

      (= (xmin b2) 2) (= (ymin b2) 2)
      (= (xmax b2) 2) (= (ymax b2) 3)

      (= (xmin b3) 1) (= (ymin b3) 3)
      (= (xmax b3) 1) (= (ymax b3) 4)

      (= (xmin b4) 3) (= (ymin b4) 4)
      (= (xmax b4) 3) (= (ymax b4) 4)

      (= (xmin b5) 2) (= (ymin b5) 4)
      (= (xmax b5) 2) (= (ymax b5) 4)

      (= (xmin b6) 1) (= (ymin b6) 1)
      (= (xmax b6) 2) (= (ymax b6) 1)

      (= (xmin b7) 3) (= (ymin b7) 1)
      (= (xmax b7) 4) (= (ymax b7) 1)

      (= (xmin b8) 5) (= (ymin b8) 1)
      (= (xmax b8) 5) (= (ymax b8) 2)

      (= (xmin b9) 5) (= (ymin b9) 3)
      (= (xmax b9) 5) (= (ymax b9) 4)

    )
  )

  (:metric minimize (total-moves))
)
