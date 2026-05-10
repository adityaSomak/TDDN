
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 j4 j5 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 14) 
    (= (capacity j2) 13) 
    (= (capacity j3) 8) 
    (= (capacity j4) 6) 
    (= (capacity j5) 1) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 1) 
    (= (contains j2) 13) 
    (= (contains j3) 4) 
    (= (contains j4) 4) 
    (= (contains j5) 0) 
) 


  (:goal
    (and 
      (= (contains j1) 13) 
      (= (contains j2) 4) 
      (= (contains j3) 0) 
      (= (contains j4) 4) 
      (= (contains j5) 1) 

    )
  )
  (:metric minimize (total-pour))
)
