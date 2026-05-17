
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 j4 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 13) 
    (= (capacity j2) 8) 
    (= (capacity j3) 5) 
    (= (capacity j4) 3) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 11) 
    (= (contains j2) 6) 
    (= (contains j3) 3) 
    (= (contains j4) 3) 
) 


  (:goal
    (and 
      (= (contains j1) 13) 
      (= (contains j2) 5) 
      (= (contains j3) 2) 
      (= (contains j4) 3) 

    )
  )
  (:metric minimize (total-pour))
)
