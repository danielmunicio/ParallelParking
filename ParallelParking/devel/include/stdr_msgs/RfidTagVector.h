// Generated by gencpp from file stdr_msgs/RfidTagVector.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_RFIDTAGVECTOR_H
#define STDR_MSGS_MESSAGE_RFIDTAGVECTOR_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <stdr_msgs/RfidTag.h>

namespace stdr_msgs
{
template <class ContainerAllocator>
struct RfidTagVector_
{
  typedef RfidTagVector_<ContainerAllocator> Type;

  RfidTagVector_()
    : rfid_tags()  {
    }
  RfidTagVector_(const ContainerAllocator& _alloc)
    : rfid_tags(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::stdr_msgs::RfidTag_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::stdr_msgs::RfidTag_<ContainerAllocator> >> _rfid_tags_type;
  _rfid_tags_type rfid_tags;





  typedef boost::shared_ptr< ::stdr_msgs::RfidTagVector_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::stdr_msgs::RfidTagVector_<ContainerAllocator> const> ConstPtr;

}; // struct RfidTagVector_

typedef ::stdr_msgs::RfidTagVector_<std::allocator<void> > RfidTagVector;

typedef boost::shared_ptr< ::stdr_msgs::RfidTagVector > RfidTagVectorPtr;
typedef boost::shared_ptr< ::stdr_msgs::RfidTagVector const> RfidTagVectorConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::stdr_msgs::RfidTagVector_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::stdr_msgs::RfidTagVector_<ContainerAllocator1> & lhs, const ::stdr_msgs::RfidTagVector_<ContainerAllocator2> & rhs)
{
  return lhs.rfid_tags == rhs.rfid_tags;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::stdr_msgs::RfidTagVector_<ContainerAllocator1> & lhs, const ::stdr_msgs::RfidTagVector_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace stdr_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::RfidTagVector_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::RfidTagVector_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::RfidTagVector_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d1ccd79235f17c9d8c9665681cfa66f7";
  }

  static const char* value(const ::stdr_msgs::RfidTagVector_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd1ccd79235f17c9dULL;
  static const uint64_t static_value2 = 0x8c9665681cfa66f7ULL;
};

template<class ContainerAllocator>
struct DataType< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stdr_msgs/RfidTagVector";
  }

  static const char* value(const ::stdr_msgs::RfidTagVector_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Rfid tag list\n"
"stdr_msgs/RfidTag[] rfid_tags\n"
"\n"
"================================================================================\n"
"MSG: stdr_msgs/RfidTag\n"
"# Rfid tag description\n"
"\n"
"string tag_id\n"
"string message\n"
"geometry_msgs/Pose2D pose # sensor pose, relative to the map origin\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose2D\n"
"# Deprecated\n"
"# Please use the full 3D pose.\n"
"\n"
"# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.\n"
"\n"
"# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.\n"
"\n"
"\n"
"# This expresses a position and orientation on a 2D manifold.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 theta\n"
;
  }

  static const char* value(const ::stdr_msgs::RfidTagVector_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.rfid_tags);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RfidTagVector_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::stdr_msgs::RfidTagVector_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::stdr_msgs::RfidTagVector_<ContainerAllocator>& v)
  {
    s << indent << "rfid_tags[]" << std::endl;
    for (size_t i = 0; i < v.rfid_tags.size(); ++i)
    {
      s << indent << "  rfid_tags[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::stdr_msgs::RfidTag_<ContainerAllocator> >::stream(s, indent + "    ", v.rfid_tags[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // STDR_MSGS_MESSAGE_RFIDTAGVECTOR_H
