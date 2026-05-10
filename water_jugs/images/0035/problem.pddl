
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 13) 
    (= (capacity j2) 12) 
    (= (capacity j3) 6) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 11) 
    (= (contains j2) 11) 
    (= (contains j3) 5) 
) 


  (:goal
    (and 
      (= (contains j1) 11) 
      (= (contains j2) 12) 
      (= (contains j3) 4) 

    )
  )
  (:metric minimize (total-pour))
)
