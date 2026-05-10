
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 j4 j5 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 11) 
    (= (capacity j2) 10) 
    (= (capacity j3) 9) 
    (= (capacity j4) 5) 
    (= (capacity j5) 4) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 1) 
    (= (contains j2) 2) 
    (= (contains j3) 5) 
    (= (contains j4) 5) 
    (= (contains j5) 4) 
) 


  (:goal
    (and 
      (= (contains j1) 9) 
      (= (contains j2) 0) 
      (= (contains j3) 0) 
      (= (contains j4) 4) 
      (= (contains j5) 4) 

    )
  )
  (:metric minimize (total-pour))
)
