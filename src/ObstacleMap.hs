-- Copyright 5-o (c) 2017
-- All rights reserved.

module ObstacleMap
    ( makeMap
    ) where

data BikeState = BikeState
    { frontWheelContactPos :: (Double, Double)
    , frontWheelContactVel :: Double 
    , steerAngle :: Double
    , leanAngle :: Double }

data InputHistory = InputHistory
    { 

data ObstacleMap = ObstacleMap Int

makeMap :: BikeState -> InputHistory -> ObstacleMap
makeMap = error "makeMap not implemented yet."
