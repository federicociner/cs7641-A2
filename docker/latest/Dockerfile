FROM federicociner/python2-base:latest

MAINTAINER Federico Ciner <federico.ciner@email.com>

ENV JYTHON_VERSION=2.7.0

# install Oracle Java
RUN apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:webupd8team/java \
    && apt-get update -y \
    && echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections \
    && apt-get install -y --no-install-recommends oracle-java8-installer oracle-java8-set-default ca-certificates \
    && rm -rf /var/cache/oracle-jdk8-installer \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# set Oracle Java as default Java
RUN update-java-alternatives -s java-8-oracle \
    && echo "export JAVA_HOME=/usr/lib/jvm/java-8-oracle" >> ~/.bashrc

# install Jython from .jar
COPY files/* /
RUN java -jar jython-installer-${JYTHON_VERSION}.jar -s -t standard -d /usr/local/jython-${JYTHON_VERSION} \
    && rm -f jython-installer-${JYTHON_VERSION}.jar \
    && ln -s /usr/local/jython-${JYTHON_VERSION}/jython /usr/local/bin/

# set Jython environment variables
ENV JYTHON_HOME=/usr/local/jython-${JYTHON_VERSION}
ENV PATH="${PATH}:${JYTHON_HOME}/bin"

# install Ant
ENV ANT_VERSION=1.10.2
RUN wget -q http://www.us.apache.org/dist//ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz && \
    tar -xzf apache-ant-${ANT_VERSION}-bin.tar.gz && \
    mv apache-ant-${ANT_VERSION} /opt/ant && \
    rm apache-ant-${ANT_VERSION}-bin.tar.gz

ENV ANT_HOME=/opt/ant
ENV PATH="${PATH}:/opt/ant/bin"

# clone ABAGAIL repo, build jar using Ant, add to Java CLASSPATH and install additional Jython packages
RUN cd /home \
    && git clone https://github.com/wollerman/ABAGAIL.git \
    && cd /home/ABAGAIL && ant \
    && echo 'export CLASSPATH=.:/home/ABAGAIL/ABAGAIL.jar' >> ~/.bashrc \
    && jython -m easy_install click
