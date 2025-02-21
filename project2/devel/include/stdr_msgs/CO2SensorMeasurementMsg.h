// Generated by gencpp from file stdr_msgs/CO2SensorMeasurementMsg.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_CO2SENSORMEASUREMENTMSG_H
#define STDR_MSGS_MESSAGE_CO2SENSORMEASUREMENTMSG_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace stdr_msgs
{
template <class ContainerAllocator>
struct CO2SensorMeasurementMsg_
{
  typedef CO2SensorMeasurementMsg_<ContainerAllocator> Type;

  CO2SensorMeasurementMsg_()
    : header()
    , co2_ppm(0.0)  {
    }
  CO2SensorMeasurementMsg_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , co2_ppm(0.0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef float _co2_ppm_type;
  _co2_ppm_type co2_ppm;





  typedef boost::shared_ptr< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> const> ConstPtr;

}; // struct CO2SensorMeasurementMsg_

typedef ::stdr_msgs::CO2SensorMeasurementMsg_<std::allocator<void> > CO2SensorMeasurementMsg;

typedef boost::shared_ptr< ::stdr_msgs::CO2SensorMeasurementMsg > CO2SensorMeasurementMsgPtr;
typedef boost::shared_ptr< ::stdr_msgs::CO2SensorMeasurementMsg const> CO2SensorMeasurementMsgConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator1> & lhs, const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.co2_ppm == rhs.co2_ppm;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator1> & lhs, const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace stdr_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ea3141a4e89d798f2735cb324ffcd870";
  }

  static const char* value(const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xea3141a4e89d798fULL;
  static const uint64_t static_value2 = 0x2735cb324ffcd870ULL;
};

template<class ContainerAllocator>
struct DataType< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stdr_msgs/CO2SensorMeasurementMsg";
  }

  static const char* value(const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Sensor measurement description\n"
"# All vectors must have the same size\n"
"\n"
"Header header\n"
"\n"
"float32 co2_ppm\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
;
  }

  static const char* value(const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.co2_ppm);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct CO2SensorMeasurementMsg_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::stdr_msgs::CO2SensorMeasurementMsg_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "co2_ppm: ";
    Printer<float>::stream(s, indent + "  ", v.co2_ppm);
  }
};

} // namespace message_operations
} // namespace ros

#endif // STDR_MSGS_MESSAGE_CO2SENSORMEASUREMENTMSG_H
