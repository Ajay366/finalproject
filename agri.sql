-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Mar 24, 2024 at 04:55 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `agri`
--

-- --------------------------------------------------------

--
-- Table structure for table `book`
--

CREATE TABLE `book` (
  `id` int(11) NOT NULL,
  `far_username` varchar(20) NOT NULL,
  `product` varchar(30) NOT NULL,
  `price` varchar(20) NOT NULL,
  `far_mobile` bigint(20) NOT NULL,
  `username` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `req_date` varchar(20) NOT NULL,
  `status` varchar(20) NOT NULL,
  `quantity` int(5) NOT NULL,
  `total` varchar(10) NOT NULL,
  `payment` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `book`
--

INSERT INTO `book` (`id`, `far_username`, `product`, `price`, `far_mobile`, `username`, `mobile`, `req_date`, `status`, `quantity`, `total`, `payment`) VALUES
(1, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', 'Processing', 0, '', ''),
(2, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', 'Delivered', 0, '', ''),
(3, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', '0', 0, '', ''),
(4, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', '', 0, '', ''),
(5, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', '', 0, '', ''),
(6, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', '', 0, '', ''),
(7, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', '', 2, '240.0', ''),
(8, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', 'Processing', 2, '240.0', 'COD'),
(9, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 19, 2024', '', 2, '240.0', 'UPI'),
(10, 'klj', 'Tomato', '120', 8148956634, 'far', 8148956634, 'March 23, 2024', '', 6, '720.0', 'UPI');

-- --------------------------------------------------------

--
-- Table structure for table `farmer`
--

CREATE TABLE `farmer` (
  `id` int(50) NOT NULL,
  `name` varchar(20) NOT NULL,
  `address` varchar(100) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `reg_date` date NOT NULL,
  `action` int(5) NOT NULL,
  `latitude` varchar(30) NOT NULL,
  `longitude` varchar(30) NOT NULL,
  KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `farmer`
--

INSERT INTO `farmer` (`id`, `name`, `address`, `mobile`, `email`, `username`, `password`, `reg_date`, `action`, `latitude`, `longitude`) VALUES
(1, 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'klj', '1234', '2024-02-24', 1, '10.8155 ', '78.69651'),
(2, 'geo', 'No.3354, Shivaram Nagar, Bikshandarkoil, Tamil Nadu', 8148956634, 'hj@gmail.com', 'geo', '1234', '2024-02-26', 1, '11.00599003', '77.56089783'),
(3, 'nani', '4/15, Srinivasa Nagar N Ext, Srirangam, Thiruvanaikoil, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'nani', '1234', '2024-02-24', 1, '10.72056961', '77.87950897'),
(4, 'ttt', '28, Kuttiambalakaranpatti, Udayanpatti, K K Nagar, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'huyyt', '1234', '2024-02-24', 1, '10.60772038', '78.42581940'),
(5, 'ioioi', 'No.3354, Shivaram Nagar, Bikshandarkoil, Tamil Nadu', 8148956634, 'yyjt@gmail.com', 'yy', '1234', '2024-02-24', 1, '10.10501003', '78.11335754');

-- --------------------------------------------------------

--
-- Table structure for table `leaf_data`
--

CREATE TABLE `leaf_data` (
  `id` int(11) NOT NULL,
  `disease` varchar(50) NOT NULL,
  `symptoms` text NOT NULL,
  `solution` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `leaf_data`
--

INSERT INTO `leaf_data` (`id`, `disease`, `symptoms`, `solution`) VALUES
(1, 'Bacterial Leaf Blight', 'Dark spots with concentric rings develop on older leaves first that touch infected soil. The surrounding leaf area may turn yellow. Affected leaves may die prematurely, exposing the fruits to sunscald. It also infects stems and fruit, presenting as black leathery spots on fruit. ', 'Pinch off leaves with leaf spots and bury them in the compost pile.\r\nIt is okay to remove up to a third of the plant''s leaves if you catch the disease early.\r\nDo not remove more than a third of the plant''s leaves.\r\nKeep leaves dry to reduce spreading the disease.'),
(2, 'Blight', 'The infection appears as small, dark spots that enlarge to 1/4-inch diameter. The spot develops a tan or gray center, and the leaves eventually wilt and fall off. Older leaves are affected first. ', 'Pinch off leaves with leaf spots and bury them in the compost pile.\r\nIt is okay to remove up to a third of the plant''s leaves if you catch the disease early.\r\nDo not remove more than a third of the plant''s leaves.\r\nKeep leaves dry to reduce spreading the disease.'),
(3, 'Blast', 'Dieback of twigs; premature leaf drop; dark staining on fruit; leaves and twigs covered in dark spores.', 'Pinch off leaves with leaf spots and bury them in the compost pile.\r\nIt is okay to remove up to a third of the plant''s leaves if you catch the disease early.'),
(4, 'Brown Spot', 'A highly contagious bacterial infection, citrus canker causes yellow halo-like lesions on fruit, leaves and twigs of citrus trees. If allowed to progress unchecked, this lemon tree problem will eventually result in dieback, fruit drop, and leaf loss.', 'Do not remove more than a third of the plant''s leaves.\r\nKeep leaves dry to reduce spreading the disease.');

-- --------------------------------------------------------

--
-- Table structure for table `post`
--

CREATE TABLE `post` (
  `id` int(50) NOT NULL,
  `product_type` varchar(20) NOT NULL,
  `product` varchar(20) NOT NULL,
  `message` varchar(100) NOT NULL,
  `price` varchar(30) NOT NULL,
  `name` varchar(30) NOT NULL,
  `address` varchar(100) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `post_date` varchar(20) NOT NULL,
  `post_time` varchar(20) NOT NULL,
  `username` varchar(20) NOT NULL,
  `post_id` int(10) NOT NULL,
  `latitude` varchar(30) NOT NULL,
  `longitude` varchar(30) NOT NULL,
  `quantity` int(10) NOT NULL,
  `image` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `post`
--

INSERT INTO `post` (`id`, `product_type`, `product`, `message`, `price`, `name`, `address`, `mobile`, `post_date`, `post_time`, `username`, `post_id`, `latitude`, `longitude`, `quantity`, `image`) VALUES
(11, 'Vegetables', 'Tomato', 'something', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '12:36 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(12, 'Vegetables', 'Tomato', 'somthing', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '12:39 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(13, 'Vegetables', 'Tomato', 'w2s', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '12:41 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(14, 'Vegetables', 'Tomato', 'wxxw', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '12:43 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(15, 'Vegetables', 'Tomato', 'wxd', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '12:45 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(16, 'Vegetables', 'Tomato', 'cdcacas', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '09:48 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(17, 'Vegetables', 'Tomato', 'Something', '120', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 19, 2024', '10:30 PM', 'klj', 0, '78.69651', '10.8155 ', 0, ''),
(18, 'Vegetables', 'Pumbkin', 'Something', '20', 'geo', 'No.3354, Shivaram Nagar, Bikshandarkoil, Tamil Nadu', 8148956634, 'March 23, 2024', '06:36 PM', 'geo', 0, '77.56089783', '11.00599003', 150, 'p3.png'),
(19, 'Vegetables', 'Pumb', 'cvbj', '20', 'man', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'March 24, 2024', '08:21 PM', 'klj', 0, '78.69651', '10.8155 ', 150, 'image_7.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `request`
--

CREATE TABLE `request` (
  `id` int(11) NOT NULL,
  `far_username` varchar(20) NOT NULL,
  `username` varchar(20) NOT NULL,
  `product` varchar(30) NOT NULL,
  `req_date` varchar(20) NOT NULL,
  `action` int(5) NOT NULL,
  `link` varchar(30) NOT NULL,
  `date` varchar(20) NOT NULL,
  `time` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `request`
--

INSERT INTO `request` (`id`, `far_username`, `username`, `product`, `req_date`, `action`, `link`, `date`, `time`) VALUES
(1, 'klj', 'far', 'Tomato', 'March 19, 2024', 1, '17771', '0000-00-00', '00:00:00'),
(2, 'klj', 'far', 'Tomato', 'March 19, 2024', 2, '12345678', '0000-00-00', '00:00:00'),
(3, 'klj', 'far', 'Tomato', 'March 19, 2024', 0, '121212', '2024-03-22', '17:36'),
(4, 'klj', 'far', 'Tomato', 'March 19, 2024', 0, '', '0000-00-00', '00:00:00'),
(5, 'klj', 'far', 'Tomato', 'March 19, 2024', 0, '', '', ''),
(6, 'klj', 'far', 'Tomato', 'March 23, 2024', 0, '', '', '');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `id` int(50) NOT NULL,
  `name` varchar(20) NOT NULL,
  `address` varchar(100) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `reg_date` date NOT NULL,
  `action` int(5) NOT NULL,
  `latitude` varchar(30) NOT NULL,
  `longitude` varchar(30) NOT NULL,
  `status` int(5) NOT NULL,
  KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `name`, `address`, `mobile`, `email`, `username`, `password`, `reg_date`, `action`, `latitude`, `longitude`, `status`) VALUES
(1, 'yuvan', '177, Chennai trunk road, Taluk, Srirangam, Thiruvanaikoil', 8148956634, 'huwaidom@gmail.com', 'yu', '1234', '2024-02-02', 1, '10.8155', '78.69651', 0),
(2, 'sankar', 'No.3354, Shivaram Nagar, Bikshandarkoil, Tamil Nadu', 9098675667, 'huwaidom@gmail.com', 'san', '1234', '2024-02-02', 1, '11.00599003', '77.56089783', 0),
(3, 'dany', '19, 34, Chandra Nagar St, Periyar Nagar, Tiruchirappalli', 9089675645, 'jai@gmail.com', 'dan', '1234', '2024-02-02', 1, '11.11540985', '77.35456085', 0),
(4, 'Madhan', 'RP2Q+XM2, Vivek Nagar, Pappakurichi Kattur, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'mad', '1234', '2024-02-02', 1, '11.07750988', '77.88362885', 0),
(5, 'Harsh', '4/15, Srinivasa Nagar N Ext, Srirangam, Thiruvanaikoil, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'harsh', '1234', '2024-02-08', 1, '10.79426003', '77.71150208', 0),
(6, 'Muthu', '28, Kuttiambalakaranpatti, Udayanpatti, K K Nagar, Tiruchirappalli', 9589859590, 'hs@gmail.com', 'muthu', '1234', '2024-02-03', 1, '11.10824966', '78.00112915', 0),
(7, 'the', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'the', '1234', '2024-02-15', 1, '10.73828030', '77.53222656', 0),
(8, 'run', '28, Kuttiambalakaranpatti, Udayanpatti, K K Nagar, Tiruchirappalli', 9589859590, 'hs@gmail.com', 'ro', '1234', '2024-02-03', 1, '10.95771027', '78.08094788', 0),
(9, 'was', '4/15, Srinivasa Nagar N Ext, Srirangam, Thiruvanaikoil, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'was', '1234', '2024-02-08', 1, '10.72056961', '77.87950897', 0),
(10, 'jack', '28, Kuttiambalakaranpatti, Udayanpatti, K K Nagar, Tiruchirappalli', 9589859590, 'hs1@gmail.com', 'jac', '1234', '2024-02-21', 1, '11.05935955', '78.13964844', 0),
(11, 'farzi', '4/15, Srinivasa Nagar N Ext, Srirangam, Thiruvanaikoil, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'far', '1234', '2024-02-24', 1, '11.15217018', '78.21205139', 0),
(12, 'io', '28, Kuttiambalakaranpatti, Udayanpatti, K K Nagar, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'io', '1234', '2024-02-24', 1, '11.14671040', '78.28996277', 0),
(13, 'david', 'Kela Mettu Street, Lakshmi Nagar, No 1 Tollgate, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'dav', '1234', '2024-02-08', 1, '10.45034027', '77.52089691', 0),
(14, 'little', 'No. 131, Ammamandapam Rd, Srirangam, Tiruchirappalli', 8148956634, 'huwaidom@gmail.com', 'dan', '1234', '2024-02-02', 1, '10.93486977', '78.41251373', 0),
(15, 'jan', '28, Kuttiambalakaranpatti, Udayanpatti, K K Nagar, Tiruchirappalli', 9589859587, 'iuu@gmail.com', 'pl', '1234', '2024-02-03', 1, '11.12417030', '78.44915771', 0),
(16, 'art', 'VPH7+F66, 1, Tollgate, Annai Nagar, No 1 Tollgate, Bikshandarkoil, Tiruchirappalli', 9087566778, 'kl@gmail.com', 'war', '1234', '2024-02-24', 1234, '10.60772038', '78.42581940', 0),
(17, 'pop', '136-2a/2b, Pudukottai Road, Gundur Village, Ramanathapuram Rd, Tiruchirappalli', 8977675690, 'haj@gmail.com', 'kop', '123', '2024-02-29', 1, '11.14968014', '78.59870148', 0),
(18, 'esaki', '4/15, Srinivasa Nagar N Ext, Srirangam, Thiruvanaikoil, Tiruchirappalli,', 6789095678, 'hgghs@gmail.com', 'uyef@gmail.com', '1234', '2024-02-14', 1, '10.53102016', '77.95018768', 0);
