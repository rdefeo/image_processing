
# libgl1-mesa-dev
# package "libgl1-mesa-dev"

apt_package "libjpeg62-dev" do
  action :install
end

apt_package "libtiff4-dev" do
  action :install
end

apt_package "libgtk2.0-dev" do
  action :install
end

apt_package "libavcodec-dev" do
  action :install
end

apt_package "libavformat-dev" do
  action :install
end

apt_package "libswscale-dev" do
  action :install
end

apt_package "libv4l-dev" do
  action :install
end

apt_package "libv4l-dev" do
  action :install
end
apt_package "python-numpy" do
  action :install
end

apt_package "python-pip" do
  action :install
end

apt_package "python-dev" do
  action :install
end

apt_package "libqt4-dev" do
  action :install
end
apt_package "libqt4-core" do
  action :install
end
apt_package "libqt4-gui" do
  action :install
end
apt_package "libqt4-xml" do
  action :install
end
apt_package "libqt4-opengl" do
  action :install
end



apt_package "yum" do
  action :install
end

remote_file "/home/vagrant/OpenCV-2.4.8.tar.bz2" do
  # source "http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.8/OpenCV-2.4.8.tar.bz2"
  source "https://github.com/Itseez/opencv/archive/2.4.8.tar.gz"
  # checksum node['nodejs']['checksum']
  mode 0644
  action :create_if_missing
end

# # --no-same-owner required overcome "Cannot change ownership" bug
# # on NFS-mounted filesystem
execute "tar -xvf OpenCV-2.4.*.tar.bz2" do
  cwd "/home/vagrant"
  creates "/home/vagrant/opencv-2.4.8"
end

directory "/home/vagrant/opencv-2.4.8/build" do
  owner "vagrant"
  group "vagrant"
  mode 00755
  action :create
end
# 
bash "build_it" do
  user "root"
  cwd "/home/vagrant/opencv-2.4.8/build"
  code <<-EOH
  cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON -D BUILD_opencv_python=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_DOCS=ON -D BUILD_EXAMPLES=ON -D WITH_OPENGL=ON ..
  make
  make install
  EOH
end

# just open file called /etc/ld.so.conf.d/opencv.conf, then insert
# 
#  /usr/local/opencv/
# then type: sudo ldconfig

# file "/etc/ld.so.conf.d/opencv.conf" do
#   owner "root"
#   group "root"
#   mode 00755
# # mode 00644
# 
#   action :create
# end

