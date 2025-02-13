FROM neo4j:5.22.0

ENV NEO4JLABS_PLUGINS '["apoc", "graph-data-science"]'

ENV NEO4J_dbms_security_procedures_unrestricted "apoc.*,gds.*"
ENV NEO4J_dbms_security_procedures_allowlist "apoc.*,gds.*"

COPY apoc-5.22.0-core.jar /plugins/
COPY neo4j-graph-data-science-2.9.0.jar /plugins/

RUN chmod -R 755 /plugins

ENV NEO4J_dbms_memory_heap_initial__size=2G
ENV NEO4J_dbms_memory_heap_max__size=2G
ENV NEO4J_dbms_memory_pagecache_size=1G

EXPOSE 7474 7687