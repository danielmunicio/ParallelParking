// Auto-generated. Do not edit!

// (in-package stdr_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let SpawnRobotResult = require('./SpawnRobotResult.js');
let actionlib_msgs = _finder('actionlib_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class SpawnRobotActionResult {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.status = null;
      this.result = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('status')) {
        this.status = initObj.status
      }
      else {
        this.status = new actionlib_msgs.msg.GoalStatus();
      }
      if (initObj.hasOwnProperty('result')) {
        this.result = initObj.result
      }
      else {
        this.result = new SpawnRobotResult();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SpawnRobotActionResult
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [status]
    bufferOffset = actionlib_msgs.msg.GoalStatus.serialize(obj.status, buffer, bufferOffset);
    // Serialize message field [result]
    bufferOffset = SpawnRobotResult.serialize(obj.result, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SpawnRobotActionResult
    let len;
    let data = new SpawnRobotActionResult(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [status]
    data.status = actionlib_msgs.msg.GoalStatus.deserialize(buffer, bufferOffset);
    // Deserialize message field [result]
    data.result = SpawnRobotResult.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += actionlib_msgs.msg.GoalStatus.getMessageSize(object.status);
    length += SpawnRobotResult.getMessageSize(object.result);
    return length;
  }

  static datatype() {
    // Returns string type for a message object
    return 'stdr_msgs/SpawnRobotActionResult';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f95a810c7b5f2579dca2ec191de2a05c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======
    
    Header header
    actionlib_msgs/GoalStatus status
    SpawnRobotResult result
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: actionlib_msgs/GoalStatus
    GoalID goal_id
    uint8 status
    uint8 PENDING         = 0   # The goal has yet to be processed by the action server
    uint8 ACTIVE          = 1   # The goal is currently being processed by the action server
    uint8 PREEMPTED       = 2   # The goal received a cancel request after it started executing
                                #   and has since completed its execution (Terminal State)
    uint8 SUCCEEDED       = 3   # The goal was achieved successfully by the action server (Terminal State)
    uint8 ABORTED         = 4   # The goal was aborted during execution by the action server due
                                #    to some failure (Terminal State)
    uint8 REJECTED        = 5   # The goal was rejected by the action server without being processed,
                                #    because the goal was unattainable or invalid (Terminal State)
    uint8 PREEMPTING      = 6   # The goal received a cancel request after it started executing
                                #    and has not yet completed execution
    uint8 RECALLING       = 7   # The goal received a cancel request before it started executing,
                                #    but the action server has not yet confirmed that the goal is canceled
    uint8 RECALLED        = 8   # The goal received a cancel request before it started executing
                                #    and was successfully cancelled (Terminal State)
    uint8 LOST            = 9   # An action client can determine that a goal is LOST. This should not be
                                #    sent over the wire by an action server
    
    #Allow for the user to associate a string with GoalStatus for debugging
    string text
    
    
    ================================================================================
    MSG: actionlib_msgs/GoalID
    # The stamp should store the time at which this goal was requested.
    # It is used by an action server when it tries to preempt all
    # goals that were requested before a certain time
    time stamp
    
    # The id provides a way to associate feedback and
    # result message with specific goal requests. The id
    # specified must be unique.
    string id
    
    
    ================================================================================
    MSG: stdr_msgs/SpawnRobotResult
    # ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======
    #result definition
    stdr_msgs/RobotIndexedMsg indexedDescription
    string message
    
    ================================================================================
    MSG: stdr_msgs/RobotIndexedMsg
    string name
    stdr_msgs/RobotMsg robot
    
    ================================================================================
    MSG: stdr_msgs/RobotMsg
    geometry_msgs/Pose2D initialPose
    
    stdr_msgs/FootprintMsg footprint
    
    stdr_msgs/LaserSensorMsg[] laserSensors
    stdr_msgs/SonarSensorMsg[] sonarSensors
    stdr_msgs/RfidSensorMsg[] rfidSensors
    stdr_msgs/CO2SensorMsg[] co2Sensors
    stdr_msgs/SoundSensorMsg[] soundSensors
    stdr_msgs/ThermalSensorMsg[] thermalSensors
    
    stdr_msgs/KinematicMsg kinematicModel
    
    ================================================================================
    MSG: geometry_msgs/Pose2D
    # Deprecated
    # Please use the full 3D pose.
    
    # In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.
    
    # If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.
    
    
    # This expresses a position and orientation on a 2D manifold.
    
    float64 x
    float64 y
    float64 theta
    
    ================================================================================
    MSG: stdr_msgs/FootprintMsg
    geometry_msgs/Point[] points
    float32 radius # for circular footprints
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: stdr_msgs/LaserSensorMsg
    # Laser Sensor description
    float32 maxAngle
    float32 minAngle
    float32 maxRange
    float32 minRange
    int32 numRays
    
    stdr_msgs/Noise noise
    
    float32 frequency
    
    string frame_id
    geometry_msgs/Pose2D pose # sensor pose, relative to robot center
    
    ================================================================================
    MSG: stdr_msgs/Noise
    bool noise
    float32 noiseMean
    float32 noiseStd
    
    ================================================================================
    MSG: stdr_msgs/SonarSensorMsg
    # Rfid sensor description
    float32 maxRange
    float32 minRange
    float32 coneAngle
    
    float32 frequency
    stdr_msgs/Noise noise
    
    string frame_id
    geometry_msgs/Pose2D pose # sensor pose, relative to robot center
    
    ================================================================================
    MSG: stdr_msgs/RfidSensorMsg
    # Rfid sensor description
    float32 maxRange
    float32 angleSpan
    float32 signalCutoff
    
    float32 frequency
    
    string frame_id
    geometry_msgs/Pose2D pose # sensor pose, relative to robot center
    
    ================================================================================
    MSG: stdr_msgs/CO2SensorMsg
    # Sensor description
    
    float32 maxRange
    float32 frequency
    string frame_id
    
    # sensor pose, relative to robot center
    geometry_msgs/Pose2D pose 
    
    ================================================================================
    MSG: stdr_msgs/SoundSensorMsg
    # Sensor description
    
    float32 maxRange
    float32 frequency
    float32 angleSpan
    string frame_id
    
    # sensor pose, relative to robot center
    geometry_msgs/Pose2D pose 
    
    ================================================================================
    MSG: stdr_msgs/ThermalSensorMsg
    # Sensor description
    
    float32 maxRange
    float32 frequency
    float32 angleSpan
    string frame_id
    
    # sensor pose, relative to robot center
    geometry_msgs/Pose2D pose 
    
    ================================================================================
    MSG: stdr_msgs/KinematicMsg
    # Container for the kinematic model parameters. The parameters selected/
    # are quite general. For a more accurate motion model a per-kinematic model
    # approach should be followed.
    # Parameters are in the form a_C_D, where C is affected by D.
    # ux is the linear speed
    # uy is the lateral speed (for omni vehicles)
    # w is the angular speed
    # g is a cofficient that directly affects the angular speed
    # For more information check the MotionController::sampleVelocities function.
    
    string type
    float32 a_ux_ux
    float32 a_ux_uy
    float32 a_ux_w
    float32 a_uy_ux
    float32 a_uy_uy
    float32 a_uy_w
    float32 a_w_ux
    float32 a_w_uy
    float32 a_w_w
    float32 a_g_ux
    float32 a_g_uy
    float32 a_g_w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SpawnRobotActionResult(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.status !== undefined) {
      resolved.status = actionlib_msgs.msg.GoalStatus.Resolve(msg.status)
    }
    else {
      resolved.status = new actionlib_msgs.msg.GoalStatus()
    }

    if (msg.result !== undefined) {
      resolved.result = SpawnRobotResult.Resolve(msg.result)
    }
    else {
      resolved.result = new SpawnRobotResult()
    }

    return resolved;
    }
};

module.exports = SpawnRobotActionResult;
